
import pandas as pd
import numpy as np
import folium
from folium import plugins
from analyzer import EnvironmentAnalyzer
from optimizer import SectorOptimizer
from rf_model import RFModel
import os
from shapely.geometry import Polygon

def main():
    print("Reading towers.xlsx...")
    try:
        df = pd.read_excel('towers.xlsx')
    except Exception as e:
        print(f"Error: {e}")
        return

    analyzer = EnvironmentAnalyzer(radius=1500)
    optimizer = SectorOptimizer(beamwidth=65, min_gap=10)
    rf_model = RFModel()

    results = []
    global_coverage_polygons = []

    # Sorting towers by potential density (priority map sum) to fulfill user request
    print("Pre-analyzing towers for sorting...")
    tower_priorities = []
    for idx, row in df.iterrows():
        p_map, ctx = analyzer.analyze_environment(row['Latitude'], row['Longitude'])
        tower_priorities.append((idx, np.sum(p_map)))

    # Sort by priority descending
    tower_priorities.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in tower_priorities]
    df = df.iloc[sorted_indices].reset_index(drop=True)

    # Initialize Map
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
    folium.TileLayer('cartodbpositron', name='Clean Map').add_to(m)
    # Add Esri Satellite
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False
    ).add_to(m)

    for idx, row in df.iterrows():
        tower_name = row['Tower_Name']
        lat, lon = row['Latitude'], row['Longitude']
        height = row.get('Height', 30)
        tilt = row.get('Tilt', 2)

        print(f"Processing {tower_name}...")

        # 1. Analyze environment with global coverage consideration
        p_map, context = analyzer.analyze_environment(lat, lon, already_covered_polygons=global_coverage_polygons)

        # 2. Optimize sectors
        azimuths, explanation = optimizer.optimize(p_map, context)

        # 3. Calculate range
        range_m = rf_model.calculate_range(context, height, tilt)

        # Add Tower Marker
        folium.Marker(
            [lat, lon],
            tooltip=f"{tower_name} ({context})",
            icon=folium.Icon(color='blue', icon='broadcast-tower', prefix='fa')
        ).add_to(m)

        tower_results = {
            'Tower_Name': tower_name,
            'Context': context,
            'Range_m': round(range_m, 2),
            'Sectors_Count': len(azimuths),
            'Explanation': explanation
        }

        for i, az in enumerate(azimuths):
            # Generate Sector Polygon (returns list of (lat, lon))
            poly_coords = rf_model.get_sector_polygon(lat, lon, az, 65, range_m)

            # Create Shapely Polygon (needs (lon, lat) for geometric operations usually,
            # but here it depends on how analyzer uses it.
            # analyzer uses poly.contains(centroid) where centroid is from OSM (lon, lat))
            # Wait, OSM Centroid is (lon, lat)? Let's check analyzer.py
            # centroid = row.geometry.centroid
            # centroid.y is lat, centroid.x is lon in analyzer.get_azimuth(lat, lon, centroid.y, centroid.x)
            # So centroid is Point(lon, lat).
            # Thus Shapely Polygon should be created with (lon, lat) points.
            shapely_poly = Polygon([(p[1], p[0]) for p in poly_coords])
            global_coverage_polygons.append(shapely_poly)

            # Map Visualization (folium needs (lat, lon))
            folium.Polygon(
                locations=poly_coords,
                color='red',
                fill=True,
                fill_color='orange',
                fill_opacity=0.4,
                weight=2,
                tooltip=f"{tower_name} - Sector {i+1}: {az}°"
            ).add_to(m)

            # Save to results
            tower_results[f'Sector_{i+1}_Azimuth'] = az

        results.append(tower_results)

    # Save Excel Report
    res_df = pd.DataFrame(results)
    res_df.to_excel('RF_Planning_Results.xlsx', index=False)
    print("Report saved to RF_Planning_Results.xlsx")

    # Save Map
    m.save('Professional_RF_Coverage_Map.html')
    print("Map saved to Professional_RF_Coverage_Map.html")

if __name__ == "__main__":
    main()