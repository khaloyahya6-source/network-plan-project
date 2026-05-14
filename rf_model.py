import numpy as np
import math

class RFModel:
    def __init__(self):
        # Base ranges (meters) for each frequency layer in "ideal" conditions
        self.base_ranges = {
            'Low': 2500,    # 700/800/900 MHz (High penetration, longest range)
            'Mid': 1200,    # 1800/2100 MHz (Medium range)
            'High': 600     # 2500/2600/3500 MHz (Short range, high capacity)
        }

        # Attenuation sensitivity (percentage range reduction per unit of clutter)
        self.attenuation_sensitivity = {
            'Low': 0.4,     # Max reduction ~40%
            'Mid': 0.5,     # Max reduction ~50%
            'High': 0.6     # Max reduction ~60%
        }

    def calculate_range(self, band_type, clutter_score, height):
        """
        Calculate effective range based on frequency band, clutter, and height.
        """
        base = self.base_ranges.get(band_type, 1000)
        sensitivity = self.attenuation_sensitivity.get(band_type, 0.5)

        # Continuous Attenuation logic
        attenuated_range = base * (1 - (clutter_score * sensitivity))

        # Height bonus: 0.5% per meter above 30m, capped at 30%
        height_bonus = 1 + min(max(height - 30, 0) * 0.005, 0.3)

        return attenuated_range * height_bonus

    def get_sector_polygon(self, lat, lon, azimuth, beamwidth, max_range):
        """Generates coordinates for a sector wedge polygon using geodesic math."""
        points = [(lat, lon)]
        R = 6371000 # Earth radius in meters

        half_bw = beamwidth / 2
        for angle in np.arange(azimuth - half_bw, azimuth + half_bw + 1, 5):
            rad_azimuth = math.radians(angle)

            # Destination point calculation using equirectangular approximation
            d_lat = (max_range * math.cos(rad_azimuth)) / R
            d_lon = (max_range * math.sin(rad_azimuth)) / (R * math.cos(math.radians(lat)))

            points.append((lat + math.degrees(d_lat), lon + math.degrees(d_lon)))

        points.append((lat, lon))
        return points
