
import numpy as np
import math

class RFModel:
    def __init__(self):
        # Base ranges in meters
        self.base_ranges = {
            'Urban': 500,
            'Suburban': 1000,
            'Rural': 1800,
            'Highway': 1500,
            'Empty/Rural': 2000
        }

    def calculate_range(self, context, height, tilt=0):
        """
        Calculate effective range based on environment, height, and tilt.
        height: meters
        tilt: degrees (assumed mechanical + electrical)
        """
        base = self.base_ranges.get(context, 800)
        
        # Height factor: Standard height is ~30m. 
        # Increase range by 1% for every meter above 30m, capped at 50%
        height_factor = 1 + min(max(height - 30, 0) * 0.01, 0.5)
        
        # Tilt factor: High tilt reduces horizontal range
        # range = range * cos(tilt) - simple approximation
        tilt_rad = math.radians(min(max(tilt, 0), 15)) # Cap tilt at 15 deg
        tilt_factor = math.cos(tilt_rad)
        
        # If tilt is very high (pointing down), it hits the ground faster
        # Effective range = height / tan(tilt) if we ignore earth curvature
        if tilt > 2:
            ground_limit = height / math.tan(math.radians(tilt))
            return min(base * height_factor * tilt_factor, ground_limit)
        
        return base * height_factor * tilt_factor

    def get_sector_polygon(self, lat, lon, azimuth, beamwidth, max_range):
        """Generate coordinates for a sector polygon."""
        points = [(lat, lon)]
        
        # Approximate meters to degrees (at equator/general)
        # 1 degree lat ~ 111,000 meters
        # 1 degree lon ~ 111,000 * cos(lat) meters
        lat_scale = 111000
        lon_scale = 111000 * math.cos(math.radians(lat))
        
        half_bw = beamwidth / 2
        # Generate arc points
        for angle in np.arange(azimuth - half_bw, azimuth + half_bw + 1, 5):
            rad = math.radians(angle)
            # Adjust angle so 0 is North (standard azimuth)
            # In math, 0 is East. So angle_math = 90 - azimuth
            math_angle = math.radians(90 - angle)
            
            dx = max_range * math.cos(math_angle)
            dy = max_range * math.sin(math_angle)
            
            p_lat = lat + (dy / lat_scale)
            p_lon = lon + (dx / lon_scale)
            points.append((p_lat, p_lon))
            
        points.append((lat, lon))
        return points

if __name__ == "__main__":
    rf = RFModel()
    r = rf.calculate_range("Urban", 45, 3)
    print(f"Calculated Range (Urban, 45m, 3deg tilt): {r:.2f}m")
    poly = rf.get_sector_polygon(33.5, 36.2, 0, 65, r)
    print(f"Polygon points count: {len(poly)}")
