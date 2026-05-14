import numpy as np
import math

class RFModel:
    def __init__(self):
        # Base ranges (meters) for each frequency layer in "ideal" open conditions
        self.base_ranges = {
            'Low': 2500,    # 700/800/900 MHz
            'Mid': 1200,    # 1800/2100 MHz
            'High': 600     # 2500/2600/3500 MHz
        }

        # Attenuation sensitivity per layer (how much clutter affects the range)
        # High bands are more sensitive to clutter
        self.attenuation_sensitivity = {
            'Low': 0.4,     # Max reduction ~40%
            'Mid': 0.5,     # Max reduction ~50%
            'High': 0.6     # Max reduction ~60% (600m -> 240m)
        }

    def calculate_range(self, band_type, clutter_score, height):
        """
        Calculate effective range based on band, clutter, and height.
        clutter_score: 0.0 (empty) to 1.0 (extremely dense urban)
        height: meters
        """
        base = self.base_ranges.get(band_type, 1000)
        sensitivity = self.attenuation_sensitivity.get(band_type, 0.5)

        # Attenuation effect: Range decreases as clutter increases
        # Using a simple linear model for now, as requested "human-like" estimation
        attenuated_range = base * (1 - (clutter_score * sensitivity))

        # Height factor: Standard height is ~30m.
        # Range improves with height (better line of sight)
        # Increase range by 0.5% for every meter above 30m, capped at 30%
        height_bonus = 1 + min(max(height - 30, 0) * 0.005, 0.3)

        return attenuated_range * height_bonus

    def get_sector_polygon(self, lat, lon, azimuth, beamwidth, max_range):
        """Generate coordinates for a sector wedge polygon."""
        points = [(lat, lon)]

        # Earth radius approximation in meters
        R = 6371000

        half_bw = beamwidth / 2
        # Generate arc points every 5 degrees
        for angle in np.arange(azimuth - half_bw, azimuth + half_bw + 1, 5):
            # Convert azimuth to math angle (0 is North, clockwise)
            # Math: 0 is East, counter-clockwise
            # math_angle = 90 - azimuth
            rad_azimuth = math.radians(angle)

            # Destination point calculation (Vincenty or simpler for small distances)
            # For RF ranges < 5km, simple equirectangular approximation is usually fine
            d_lat = (max_range * math.cos(rad_azimuth)) / R
            d_lon = (max_range * math.sin(rad_azimuth)) / (R * math.cos(math.radians(lat)))

            p_lat = lat + math.degrees(d_lat)
            p_lon = lon + math.degrees(d_lon)
            points.append((p_lat, p_lon))

        points.append((lat, lon))
        return points

if __name__ == "__main__":
    rf = RFModel()
    # Test High Band in dense urban
    r_high = rf.calculate_range("High", 1.0, 30)
    print(f"High Band (Dense Urban): {r_high:.2f}m")

    # Test Low Band in rural
    r_low = rf.calculate_range("Low", 0.1, 45)
    print(f"Low Band (Rural, 45m): {r_low:.2f}m")
