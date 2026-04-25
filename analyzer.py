import osmnx as ox
import numpy as np
import pandas as pd
from shapely.geometry import Point
import math

class EnvironmentAnalyzer:
    def __init__(self, radius=1000):
        self.radius = radius
        # Define priority weights for different OSM tags
        self.weights = {
            'building': {
                'residential': 10,
                'commercial': 15,
                'retail': 15,
                'industrial': 8,
                'school': 20,
                'hospital': 20,
                'mosque': 18,
                'church': 18,
                'yes': 5 # Default building
            },
            'amenity': {
                'marketplace': 25,
                'school': 20,
                'university': 15,
                'place_of_worship': 18,
                'hospital': 20,
                'mall': 25,
                'bus_station': 15
            },
            'highway': {
                'motorway': 12,
                'primary': 10,
                'secondary': 8,
                'tertiary': 5
            },
            'landuse': {
                'residential': 10,
                'commercial': 15,
                'industrial': 8,
                'retail': 15
            }
        }

    def get_azimuth(self, lat1, lon1, lat2, lon2):
        """Calculate the bearing between two points."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        d_lon = lon2 - lon1
        y = math.sin(d_lon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
        bearing = math.atan2(y, x)
        return (math.degrees(bearing) + 360) % 360

    def analyze_environment(self, lat, lon, already_covered_polygons=None):
        print(f"Analyzing environment at ({lat}, {lon})...")
        if already_covered_polygons is None:
            already_covered_polygons = []
        try:
            # Fetch features from OSM
            tags = {
                'building': True,
                'amenity': True,
                'highway': True,
                'landuse': True
            }
            gdf = ox.features_from_point((lat, lon), tags=tags, dist=self.radius)
        except Exception as e:
            print(f"Warning: Could not fetch OSM data: {e}")
            return np.zeros(360), "Empty/Rural"

        if gdf.empty:
            return np.zeros(360), "Empty/Rural"

        priority_map = np.zeros(360)

        # Determine overall context
        total_buildings = len(gdf[gdf['building'].notna()]) if 'building' in gdf.columns else 0
        total_roads = len(gdf[gdf['highway'].notna()]) if 'highway' in gdf.columns else 0

        context = "Urban" if total_buildings > 50 else "Suburban" if total_buildings > 10 else "Rural"
        if total_roads > 5 and total_buildings < 5:
            context = "Highway"

        for _, row in gdf.iterrows():
            # Get center of the feature
            centroid = row.geometry.centroid

            # Check if this feature is already covered by existing sectors
            is_covered = False
            for poly in already_covered_polygons:
                if poly.contains(centroid):
                    is_covered = True
                    # print(f"DEBUG: Feature {row.get('name', 'unnamed')} is covered by an existing sector.")
                    break

            if is_covered:
                # Optionally, instead of skipping, we could reduce priority
                # but user said "يفضل الاستبعاد نهائي" (prefer total exclusion)
                continue

            azimuth = self.get_azimuth(lat, lon, centroid.y, centroid.x)

            weight = 1 # Default weight

            # Check tags for weights
            for tag_type, values in self.weights.items():
                if tag_type in row and pd.notna(row[tag_type]):
                    tag_value = row[tag_type]
                    weight = max(weight, values.get(tag_value, values.get('yes', 1)))

            # Distribute weight around the azimuth (accounting for size)
            # For simplicity, we use a fixed spread or calculate based on distance
            dist = Point(lon, lat).distance(centroid) # This is crude degree distance
            # Better: use a small spread
            spread = 5
            for i in range(-spread, spread + 1):
                angle = int((azimuth + i) % 360)
                priority_map[angle] += weight

        # Smooth the priority map
        priority_map = np.convolve(priority_map, np.ones(11)/11, mode='same')

        return priority_map, context

if __name__ == "__main__":
    analyzer = EnvironmentAnalyzer()
    p_map, ctx = analyzer.analyze_environment(33.5138, 36.2765) # Damascus
    print(f"Context: {ctx}")
    print(f"Max Priority at: {np.argmax(p_map)} degrees")
