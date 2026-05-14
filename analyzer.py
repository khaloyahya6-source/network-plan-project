import osmnx as ox
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import math

class EnvironmentAnalyzer:
    def __init__(self, search_radius=2500):
        self.search_radius = search_radius
        # Priority weights for clutter calculation
        self.clutter_weights = {
            'building': 1.0,
            'amenity': 0.8,
            'landuse_residential': 0.7,
            'landuse_industrial': 0.6,
            'highway': 0.4
        }

    def get_azimuth(self, lat1, lon1, lat2, lon2):
        """Calculate the bearing between two points."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        d_lon = lon2 - lon1
        y = math.sin(d_lon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
        bearing = math.atan2(y, x)
        return (math.degrees(bearing) + 360) % 360

    def fetch_osm_data(self, lat, lon):
        """Fetch OSM features around the site."""
        try:
            tags = {
                'building': True,
                'amenity': True,
                'highway': True,
                'landuse': ['residential', 'industrial', 'commercial', 'retail']
            }
            gdf = ox.features_from_point((lat, lon), tags=tags, dist=self.search_radius)
            return gdf
        except Exception as e:
            print(f"OSM Fetch Error: {e}")
            return pd.DataFrame()

    def get_sector_clutter_score(self, lat, lon, azimuth, beamwidth, max_range, gdf):
        """
        Calculate a continuous Clutter/Density Score (0-1) for a specific sector wedge.
        """
        if gdf.empty:
            return 0.0

        half_bw = beamwidth / 2
        min_angle = (azimuth - half_bw) % 360
        max_angle = (azimuth + half_bw) % 360

        total_weight = 0

        # Max weight estimation for normalization (based on a typical dense urban sector)
        # This is a "human-like" heuristic normalization
        max_theoretical_weight = 50

        for _, row in gdf.iterrows():
            centroid = row.geometry.centroid
            # Distance check (crude)
            dist = self.haversine(lat, lon, centroid.y, centroid.x)
            if dist > max_range:
                continue

            # Azimuth check
            target_az = self.get_azimuth(lat, lon, centroid.y, centroid.x)

            # Handle angle wrap-around
            is_in_wedge = False
            if min_angle < max_angle:
                is_in_wedge = min_angle <= target_az <= max_angle
            else: # Wrap around North
                is_in_wedge = target_az >= min_angle or target_az <= max_angle

            if is_in_wedge:
                weight = 0
                if 'building' in row and pd.notna(row['building']): weight = self.clutter_weights['building']
                elif 'amenity' in row and pd.notna(row['amenity']): weight = self.clutter_weights['amenity']
                elif 'landuse' in row and pd.notna(row['landuse']):
                    if row['landuse'] == 'industrial': weight = self.clutter_weights['landuse_industrial']
                    else: weight = self.clutter_weights['landuse_residential']
                elif 'highway' in row and pd.notna(row['highway']): weight = self.clutter_weights['highway']

                # Weight decays with distance (Inverse Square Law like)
                distance_factor = 1 / (1 + (dist / 500)**2)
                total_weight += weight * distance_factor

        clutter_score = min(total_weight / max_theoretical_weight, 1.0)
        return clutter_score

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def classify_site(self, lat, lon, gdf):
        """Identify if a site is Highway, Urban, or Rural."""
        if gdf.empty:
            return "Rural"

        # Highway check: site near a motorway and low building count
        near_motorway = False
        if 'highway' in gdf.columns:
            motorways = gdf[gdf['highway'].isin(['motorway', 'primary'])]
            for _, row in motorways.iterrows():
                if self.haversine(lat, lon, row.geometry.centroid.y, row.geometry.centroid.x) < 150:
                    near_motorway = True
                    break

        building_count = len(gdf[gdf['building'].notna()]) if 'building' in gdf.columns else 0

        if near_motorway and building_count < 10:
            return "Highway"
        elif building_count > 40:
            return "Urban"
        else:
            return "Rural"

    def detect_uncovered_clusters(self, lat, lon, sector_polygons, gdf):
        """Find if there are isolated clusters not covered by current sectors."""
        if gdf.empty or 'building' not in gdf.columns:
            return False

        buildings = gdf[gdf['building'].notna()]
        uncovered_count = 0

        for _, row in buildings.iterrows():
            centroid = row.geometry.centroid
            # We only care about buildings within 1.5km
            if self.haversine(lat, lon, centroid.y, centroid.x) > 1500:
                continue

            is_covered = False
            p = Point(centroid.x, centroid.y) # Point takes (x, y) which is (lon, lat)
            # Actually sector_polygons should be in (lon, lat) for Shapely
            for poly in sector_polygons:
                if poly.contains(p):
                    is_covered = True
                    break

            if not is_covered:
                uncovered_count += 1

        return uncovered_count >= 5 # Cluster trigger: 5+ buildings

if __name__ == "__main__":
    analyzer = EnvironmentAnalyzer()
    # Test coordinates
    lat, lon = 33.5138, 36.2765
    data = analyzer.fetch_osm_data(lat, lon)
    cls = analyzer.classify_site(lat, lon, data)
    print(f"Site Classification: {cls}")
    score = analyzer.get_sector_clutter_score(lat, lon, 0, 65, 1000, data)
    print(f"North Sector Clutter Score: {score:.2f}")
