import numpy as np
from config import get_rf_params, THRESHOLDS

class PhysicsEngine:
    def __init__(self, grid_data):
        self.grid_data = grid_data
        self.lats = grid_data['lat_grid']
        self.lons = grid_data['lon_grid']
        self.elev = grid_data['elevation']
        self.density = grid_data['clutter_density']

        # Constants
        self.C_LIGHT = 3e8
        self.NOISE_FLOOR = THRESHOLDS['SINR']['NoiseFloor']

    def calculate_vectors(self, site_lat, site_lon, site_height):
        # Local flat-earth projection
        # dy = (lat2 - lat1) * 111000
        # dx = (lon2 - lon1) * 111000 * cos(lat1)

        lat_rad = np.radians(site_lat)
        dy = (self.lats - site_lat) * 111000
        dx = (self.lons - site_lon) * 111000 * np.cos(lat_rad)

        dist_2d = np.sqrt(dx**2 + dy**2)
        dist_2d = np.maximum(dist_2d, 1.0) # Avoid division by zero

        # Azimuth (heading from site to pixel)
        # np.arctan2(x, y) returns angle from y-axis
        azimuths = np.degrees(np.arctan2(dx, dy)) % 360

        # Elevation Angle
        # h_pixel = elev + mobile_height (assume 1.5m)
        h_diff = (self.elev + 1.5) - (self.grid_data['elevation'][self.find_nearest_idx(site_lat, site_lon)] + site_height)
        # theta is angle from horizon. Negative is down.
        elevation_angles = np.degrees(np.arctan2(h_diff, dist_2d))

        return dist_2d, azimuths, elevation_angles

    def find_nearest_idx(self, lat, lon):
        dist = (self.lats - lat)**2 + (self.lons - lon)**2
        return np.unravel_index(np.argmin(dist), dist.shape)

    def get_path_loss(self, dist_m, freq_mhz, h_b, density):
        dist_km = dist_m / 1000.0

        if freq_mhz < 2000:
            # Modified COST231 Hata
            # L = 46.3 + 33.9log10(f) - 13.82log10(hb) - a(hm) + (44.9 - 6.55log10(hb))log10(d)
            # a(hm) for urban: (1.1log10(f)-0.7)hm - (1.56log10(f)-0.8)
            a_hm = (1.1 * np.log10(freq_mhz) - 0.7) * 1.5 - (1.56 * np.log10(freq_mhz) - 0.8)
            L = 46.3 + 33.9 * np.log10(freq_mhz) - 13.82 * np.log10(h_b) - a_hm + \
                (44.9 - 6.55 * np.log10(h_b)) * np.log10(dist_km)

            # Adjust for density (Clutter Loss)
            # Rural offset
            rural_offset = -4.78 * (np.log10(freq_mhz))**2 + 18.33 * np.log10(freq_mhz) - 40.94
            # Interpolate between Urban and Rural based on density (0 to 1)
            L = L + (1 - density) * rural_offset
        else:
            # 3GPP TR 38.901 UMa (Simplified NLOS)
            # PL = 32.4 + 20log10(d) + 20log10(fc)
            # For NLOS: PL = 13.54 + 39.08 log10(d) + 20 log10(fc) - 0.6(hm - 1.5)
            # Here d is 3D distance, but we use 2D as approximation for large d
            PL_LOS = 32.4 + 20 * np.log10(dist_m) + 20 * np.log10(freq_mhz / 1000.0)
            PL_NLOS = 13.54 + 39.08 * np.log10(dist_m) + 20 * np.log10(freq_mhz / 1000.0)

            # Dynamic weighting based on density
            L = PL_LOS * (1 - density) + PL_NLOS * density

        return L

    def antenna_gain_3d(self, azimuth_offset, elevation_offset, params, tilt):
        # azimuth_offset: pixel_azimuth - sector_azimuth
        # elevation_offset: pixel_elevation (from site perspective)
        # tilt: mechanical + electrical tilt (positive is down)

        # Horizontal Pattern
        phi = azimuth_offset
        phi = (phi + 180) % 360 - 180 # Map to [-180, 180]
        A_H = -np.minimum(12 * (phi / params['hbw'])**2, params['front_to_back'])

        # Vertical Pattern
        # Vertical beam is centered at -tilt
        theta = elevation_offset
        theta_tilt = -tilt # Site looking down
        A_V = -np.minimum(12 * ((theta - theta_tilt) / params['vbw'])**2, params['front_to_back'])

        # Combined 3GPP pattern
        gain_combined = params['antenna_gain_dbi'] - np.minimum(-(A_H + A_V), params['front_to_back'])
        return gain_combined

    def compute_rsrp(self, tower_row, azimuth, tilt, freq_mhz):
        params = get_rf_params(freq_mhz)
        dist, az_map, el_map = self.calculate_vectors(tower_row['Lat'], tower_row['Lon'], tower_row['Total_Height_m'])

        # Path Loss
        pl = self.get_path_loss(dist, freq_mhz, tower_row['Total_Height_m'], self.density)

        # Antenna Gain
        gain = self.antenna_gain_3d(az_map - azimuth, el_map, params, tilt)

        # RSRP = TxPower + Gain - PathLoss
        rsrp = params['tx_power_dbm'] + gain - pl
        return rsrp

    def compute_global_sinr(self, rsrp_matrices):
        """
        rsrp_matrices: dict or list of RSRP matrices for all sectors in a tech layer
        SINR = S / (I + N)
        """
        # Convert dBm to Watts
        # P(W) = 10^((P(dBm)-30)/10)

        power_watts = []
        for mat in rsrp_matrices:
            # Clip extremely low values to avoid overflow
            clipped = np.maximum(mat, -150)
            power_watts.append(10**((clipped - 30) / 10))

        power_stack = np.stack(power_watts)
        total_power = np.sum(power_stack, axis=0)

        sinr_matrices = []
        noise_watts = 10**((self.NOISE_FLOOR - 30) / 10)

        for p_s in power_watts:
            interference = total_power - p_s
            sinr_w = p_s / (interference + noise_watts)
            sinr_db = 10 * np.log10(np.maximum(sinr_w, 1e-15))
            sinr_matrices.append(sinr_db)

        return sinr_matrices

    def get_sector_polygon(self, lat, lon, azimuth, beamwidth, max_range_m):
        """Generates coordinates for a sector wedge polygon."""
        points = [(lon, lat)]

        # Approximate degrees
        lat_scale = 111000
        lon_scale = 111000 * np.cos(np.radians(lat))

        half_bw = beamwidth / 2
        # Generate arc points
        for angle in np.arange(azimuth - half_bw, azimuth + half_bw + 1, 5):
            math_angle = np.radians(90 - angle)

            dx = max_range_m * np.cos(math_angle)
            dy = max_range_m * np.sin(math_angle)

            p_lat = lat + (dy / lat_scale)
            p_lon = lon + (dx / lon_scale)
            points.append((p_lon, p_lat))

        points.append((lon, lat))
        return points
