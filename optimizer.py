import numpy as np
from shapely.geometry import Polygon, Point
import math
import pandas as pd

class SectorOptimizer:
    def __init__(self, beamwidth=65, min_gap=15):
        self.beamwidth = beamwidth
        self.min_gap = min_gap

    def optimize_azimuths(self, lat, lon, site_type, analyzer, rf_model, bands, osm_data, global_coverage_polygons):
        """
        Determine optimal azimuths using white-space filling logic.
        """
        if site_type == "Highway":
            # 1. Find nearest highway bearing
            road_az = self._get_nearest_road_azimuth(lat, lon, osm_data, analyzer)
            azimuths = [road_az, (road_az + 180) % 360]

            # Check for 3rd sector trigger (residential off-highway)
            if analyzer.detect_uncovered_clusters(lat, lon, [], osm_data):
                 # Find best azimuth for 3rd sector that doesn't overlap highway ones
                 best_3rd = self._find_best_whitespace_azimuth(lat, lon, azimuths, analyzer, rf_model, bands, osm_data, global_coverage_polygons)
                 if best_3rd is not None:
                     azimuths.append(best_3rd)
            return sorted(azimuths)

        # Regular/Rural logic
        azimuths = []
        target_sectors = 3 # Default

        # Iteratively add sectors
        for i in range(4):
            if i < 3 or (i == 3 and analyzer.detect_uncovered_clusters(lat, lon, self._get_shapely_polys(lat, lon, azimuths, rf_model, bands), osm_data)):
                best_az = self._find_best_whitespace_azimuth(lat, lon, azimuths, analyzer, rf_model, bands, osm_data, global_coverage_polygons)
                if best_az is not None:
                    azimuths.append(best_az)
                else:
                    break
            else:
                break

        return sorted(azimuths)

    def _get_nearest_road_azimuth(self, lat, lon, osm_data, analyzer):
        if osm_data.empty or 'highway' not in osm_data.columns:
            return 0
        roads = osm_data[osm_data['highway'].isin(['motorway', 'primary'])]
        if roads.empty: return 0

        # Get nearest road segment
        nearest_road = roads.iloc[0] # Simplification
        centroid = nearest_road.geometry.centroid
        return analyzer.get_azimuth(lat, lon, centroid.y, centroid.x)

    def _find_best_whitespace_azimuth(self, lat, lon, existing_azimuths, analyzer, rf_model, bands, osm_data, global_polys):
        best_az = None
        max_whitespace_score = -1

        # Step every 5 degrees
        for az in range(0, 360, 5):
            # 1. Check intra-site gap constraint
            is_valid = True
            for ex_az in existing_azimuths:
                diff = abs(az - ex_az)
                if diff > 180: diff = 360 - diff
                if diff < (self.beamwidth + self.min_gap):
                    is_valid = False
                    break
            if not is_valid: continue

            # 2. Calculate "White Space" Score
            # Score = (Sum of uncovered priority targets in this wedge)
            score = self._calculate_wedge_whitespace_score(lat, lon, az, analyzer, rf_model, bands, osm_data, global_polys)

            if score > max_whitespace_score:
                max_whitespace_score = score
                best_az = az

        return best_az if max_whitespace_score > 0 else None

    def _calculate_wedge_whitespace_score(self, lat, lon, azimuth, analyzer, rf_model, bands, osm_data, global_polys):
        # We use the Low band for the largest possible white space search
        max_range = rf_model.calculate_range("Low", 0.5, 30)

        score = 0
        if osm_data.empty: return 0

        half_bw = self.beamwidth / 2
        min_a = (azimuth - half_bw) % 360
        max_a = (azimuth + half_bw) % 360

        for _, row in osm_data.iterrows():
            centroid = row.geometry.centroid

            # 1. Check if it's already covered by other towers
            is_globally_covered = False
            p = Point(centroid.x, centroid.y)
            for g_poly in global_polys:
                if g_poly.contains(p):
                    is_globally_covered = True
                    break
            if is_globally_covered: continue

            # 2. Check if it's in this wedge
            target_az = analyzer.get_azimuth(lat, lon, centroid.y, centroid.x)
            in_wedge = False
            if min_a < max_a: in_wedge = min_a <= target_az <= max_a
            else: in_wedge = target_az >= min_a or target_az <= max_a

            if in_wedge:
                # Priority targets (schools, hospitals) get more score
                weight = 1
                if 'amenity' in row and pd.notna(row['amenity']): weight = 5
                score += weight

        return score

    def _get_shapely_polys(self, lat, lon, azimuths, rf_model, bands):
        polys = []
        for az in azimuths:
            # Use Low band for the primary coverage check
            r = rf_model.calculate_range("Low", 0.5, 30)
            coords = rf_model.get_sector_polygon(lat, lon, az, self.beamwidth, r)
            polys.append(Polygon([(p[1], p[0]) for p in coords]))
        return polys
