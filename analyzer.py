import osmnx as ox
import numpy as np
import pandas as pd
from shapely.geometry import Point
import math

class EnvironmentAnalyzer:
    def __init__(self, search_radius=2500):
        self.search_radius = search_radius
        self.clutter_weights = {
            'building': 1.0,
            'amenity': 0.8,
            'landuse': 0.7,
            'highway': 0.4
        }

    def get_azimuth(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        d_lon = lon2 - lon1
        y = math.sin(d_lon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
        return (math.degrees(math.atan2(y, x)) + 360) % 360

    def fetch_osm_data(self, lat, lon):
        """Fetches 2D environment features from OpenStreetMap."""
        try:
            tags = {'building': True, 'amenity': True, 'highway': True, 'landuse': True}
            return ox.features_from_point((lat, lon), tags=tags, dist=self.search_radius)
        except:
            return pd.DataFrame()

    def get_sector_clutter_score(self, lat, lon, azimuth, beamwidth, max_range, gdf):
        """Calculates a continuous clutter score (0-1) for a specific sector wedge."""
        if gdf.empty: return 0.0
        min_a, max_a = (azimuth - beamwidth/2) % 360, (azimuth + beamwidth/2) % 360
        total_weight, max_theoretical_weight = 0, 50

        for _, row in gdf.iterrows():
            centroid = row.geometry.centroid
            dist = self.haversine(lat, lon, centroid.y, centroid.x)
            if dist > max_range: continue

            target_az = self.get_azimuth(lat, lon, centroid.y, centroid.x)
            in_wedge = (min_a <= target_az <= max_a) if min_a < max_a else (target_az >= min_a or target_az <= max_a)

            if in_wedge:
                # Buildings contribute the most to attenuation
                weight = 1.0 if 'building' in row else 0.4
                # Weight decays with distance from the tower
                total_weight += weight / (1 + (dist / 500)**2)

        return min(total_weight / max_theoretical_weight, 1.0)

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi, dlambda = math.radians(lat2-lat1), math.radians(lon2-lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def classify_site(self, lat, lon, gdf):
        """Intelligently classifies the site as Highway, Urban, or Rural."""
        if gdf.empty: return "Rural"
        build_cnt = len(gdf[gdf['building'].notna()]) if 'building' in gdf.columns else 0
        is_hwy = False
        if 'highway' in gdf.columns:
            is_hwy = any(self.haversine(lat, lon, r.geometry.centroid.y, r.geometry.centroid.x) < 150
                         for _, r in gdf[gdf['highway'].isin(['motorway', 'primary'])].iterrows())

        return "Highway" if is_hwy and build_cnt < 15 else "Urban" if build_cnt > 40 else "Rural"

    def detect_uncovered_clusters(self, lat, lon, sector_polygons, gdf):
        """Checks for isolated populated clusters that require an additional sector."""
        if gdf.empty or 'building' not in gdf.columns: return False
        uncovered = 0
        for _, row in gdf[gdf['building'].notna()].iterrows():
            if self.haversine(lat, lon, row.geometry.centroid.y, row.geometry.centroid.x) > 1500: continue
            if not any(poly.contains(Point(row.geometry.centroid.x, row.geometry.centroid.y)) for poly in sector_polygons):
                uncovered += 1
        return uncovered >= 5 # Cluster trigger
