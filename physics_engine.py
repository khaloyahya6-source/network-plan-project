import numpy as np
from config import get_tech_specs, THRESHOLDS, METERS_PER_DEGREE

class PhysicsEngine:
    def __init__(self, grid_data):
        self.grid_data = grid_data
        self.lats = grid_data['lat_grid']
        self.lons = grid_data['lon_grid']
        self.elev = grid_data['elevation']
        self.density = grid_data['clutter_density']

        self.NOISE_FLOOR = THRESHOLDS['SINR']['NoiseFloor']

    def calculate_vectors(self, site_lat, site_lon, site_height):
        # Professional Coordinate Scaling
        lat_rad = np.radians(site_lat)
        # dy = delta_lat * 111320
        dy = (self.lats - site_lat) * METERS_PER_DEGREE
        # dx = delta_lon * 111320 * cos(lat)
        dx = (self.lons - site_lon) * (METERS_PER_DEGREE * np.cos(lat_rad))

        dist_2d = np.sqrt(dx**2 + dy**2)
        dist_2d = np.maximum(dist_2d, 1.0)

        azimuths = np.degrees(np.arctan2(dx, dy)) % 360

        idx = self.find_nearest_idx(site_lat, site_lon)
        site_ground_elev = self.elev[idx]

        # Elevation Angle to each pixel
        h_diff = (self.elev + 1.5) - (site_ground_elev + site_height)
        elevation_angles = np.degrees(np.arctan2(h_diff, dist_2d))

        return dist_2d, azimuths, elevation_angles

    def find_nearest_idx(self, lat, lon):
        dist = (self.lats - lat)**2 + (self.lons - lon)**2
        return np.unravel_index(np.argmin(dist), dist.shape)

    def get_path_loss(self, dist_m, freq_mhz, h_b, density, penetration_loss):
        dist_km = dist_m / 1000.0

        if freq_mhz < 2000:
            # COST231-Hata Model
            a_hm = (1.1 * np.log10(freq_mhz) - 0.7) * 1.5 - (1.56 * np.log10(freq_mhz) - 0.8)
            L = 46.3 + 33.9 * np.log10(freq_mhz) - 13.82 * np.log10(h_b) - a_hm + \
                (44.9 - 6.55 * np.log10(h_b)) * np.log10(dist_km)
        else:
            # 3GPP TR 38.901 Urban Macro (UMa) NLOS
            L = 13.54 + 39.08 * np.log10(dist_m) + 20 * np.log10(freq_mhz / 1000.0)

        # Obstacle & Penetration Attenuation
        # Attenuation scales with local clutter density and frequency-dependent penetration factor
        clutter_loss = density * penetration_loss
        return L + clutter_loss

    def antenna_gain_3d(self, azimuth_offset, elevation_offset, specs, tilt):
        # 3GPP 3D Radiation Pattern
        phi = (azimuth_offset + 180) % 360 - 180
        A_H = -np.minimum(12 * (phi / specs['hbw'])**2, 25)

        theta = elevation_offset
        theta_tilt = -tilt # Down-tilt is negative elevation from horizon
        A_V = -np.minimum(12 * ((theta - theta_tilt) / specs['vbw'])**2, 20)

        gain_combined = specs['ant_gain_dbi'] - np.minimum(-(A_H + A_V), 25)
        return gain_combined

    def compute_rsrp(self, tower_row, azimuth, tilt, tech_str):
        specs = get_tech_specs(tech_str)
        dist, az_map, el_map = self.calculate_vectors(tower_row['Lat'], tower_row['Lon'], tower_row['Total_Height_m'])
        pl = self.get_path_loss(dist, specs['freq_mhz'], tower_row['Total_Height_m'], self.density, specs['penetration_loss_db'])
        gain = self.antenna_gain_3d(az_map - azimuth, el_map, specs, tilt)
        rsrp = specs['tx_power_dbm'] + gain - pl
        return rsrp

    def compute_global_sinr(self, rsrp_matrices):
        power_watts = []
        for mat in rsrp_matrices:
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

    def calculate_cell_edge_range(self, tower_row, azimuth, tilt, tech_str):
        """Calculates dynamic radius where RSRP hits -95 dBm, capped at 1500m."""
        specs = get_tech_specs(tech_str)
        threshold = -95.0 # Cell Edge target for KML and overshooting

        # Strict constraints: 10m to 1500m
        low = 10
        high = 1500

        s_height = tower_row['Total_Height_m']
        idx = self.find_nearest_idx(tower_row['Lat'], tower_row['Lon'])
        local_density = self.density[idx]

        for _ in range(12):
            mid = (low + high) / 2
            pl = self.get_path_loss(mid, specs['freq_mhz'], s_height, local_density, specs['penetration_loss_db'])
            h_diff = -s_height
            el_angle = np.degrees(np.arctan2(h_diff, mid))
            gain = self.antenna_gain_3d(0, el_angle, specs, tilt)
            rsrp = specs['tx_power_dbm'] + gain - pl

            if rsrp > threshold: low = mid
            else: high = mid

        return low

    def get_sector_polygon(self, lat, lon, azimuth, beamwidth, max_range_m):
        """Generates Lat/Lon coordinates for a sector wedge with correct meter-to-degree scaling."""
        points = [(lon, lat)]
        # Correct Mathematical Scaling
        lat_scale = METERS_PER_DEGREE
        lon_scale = METERS_PER_DEGREE * np.cos(np.radians(lat))

        half_bw = beamwidth / 2
        for angle in np.arange(azimuth - half_bw, azimuth + half_bw + 1, 5):
            math_angle = np.radians(90 - angle)
            dx = max_range_m * np.cos(math_angle)
            dy = max_range_m * np.sin(math_angle)

            p_lat = lat + (dy / lat_scale)
            p_lon = lon + (dx / lon_scale)
            points.append((p_lon, p_lat))

        points.append((lon, lat))
        return points
