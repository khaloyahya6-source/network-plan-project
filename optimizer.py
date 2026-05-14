import numpy as np
from shapely.geometry import Polygon, Point
import math
import pandas as pd

class SectorOptimizer:
    def __init__(self, beamwidth=65, min_gap=15):
        self.beamwidth = beamwidth
        self.min_gap = min_gap

    def optimize_azimuths(self, lat, lon, site_type, analyzer, rf_model, osm_data, global_polys):
        """Determines optimal azimuths using white-space filling and strict gap rules."""
        if site_type == "Highway":
            road_az = self._get_nearest_road_azimuth(lat, lon, osm_data, analyzer)
            azimuths = [road_az, (road_az + 180) % 360]
            if analyzer.detect_uncovered_clusters(lat, lon, [], osm_data):
                 best_3rd = self._find_best_whitespace_azimuth(lat, lon, azimuths, analyzer, rf_model, osm_data, global_polys)
                 if best_3rd is not None: azimuths.append(best_3rd)
            return sorted(azimuths)

        azimuths = []
        for i in range(4):
            # Iteratively add sectors, only adding the 4th if an uncovered cluster exists
            if i < 3 or (i == 3 and analyzer.detect_uncovered_clusters(lat, lon, self._get_shapely_polys(lat, lon, azimuths, rf_model), osm_data)):
                best_az = self._find_best_whitespace_azimuth(lat, lon, azimuths, analyzer, rf_model, osm_data, global_polys)
                if best_az is not None: azimuths.append(best_az)
                else: break
        return sorted(azimuths)

    def _get_nearest_road_azimuth(self, lat, lon, osm_data, analyzer):
        if osm_data.empty or 'highway' not in osm_data.columns: return 0
        roads = osm_data[osm_data['highway'].isin(['motorway', 'primary'])]
        if roads.empty: return 0
        return analyzer.get_azimuth(lat, lon, roads.iloc[0].geometry.centroid.y, roads.iloc[0].geometry.centroid.x)

    def _find_best_whitespace_azimuth(self, lat, lon, existing, analyzer, rf_model, osm_data, global_polys):
        best_az, max_score = None, -1
        for az in range(0, 360, 5):
            # Strict 10-15 degree gap enforcement
            if any(min(abs(az-ex), 360-abs(az-ex)) < (self.beamwidth + self.min_gap) for ex in existing): continue
            score = self._calculate_score(lat, lon, az, analyzer, rf_model, osm_data, global_polys)
            if score > max_score: max_score, best_az = score, az
        return best_az if max_score > 0 else None

    def _calculate_score(self, lat, lon, az, analyzer, rf_model, osm_data, global_polys):
        """Scores a candidate azimuth based on uncovered OSM features."""
        max_r = rf_model.calculate_range("Low", 0.5, 30)
        score, min_a, max_a = 0, (az - self.beamwidth/2) % 360, (az + self.beamwidth/2) % 360
        for _, row in osm_data.iterrows():
            p = Point(row.geometry.centroid.x, row.geometry.centroid.y)
            # Global Anti-Overlap: Avoid areas covered by other towers
            if any(g.contains(p) for g in global_polys): continue
            t_az = analyzer.get_azimuth(lat, lon, row.geometry.centroid.y, row.geometry.centroid.x)
            if (min_a <= t_az <= max_a) if min_a < max_a else (t_az >= min_a or t_az <= max_a):
                score += 5 if 'amenity' in row else 1
        return score

    def _get_shapely_polys(self, lat, lon, azimuths, rf_model):
        polys = []
        for az in azimuths:
            r = rf_model.calculate_range("Low", 0.5, 30)
            coords = rf_model.get_sector_polygon(lat, lon, az, self.beamwidth, r)
            polys.append(Polygon([(p[1], p[0]) for p in coords]))
        return polys
