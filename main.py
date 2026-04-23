
import pandas as pd
import folium
from folium import plugins
from analyzer import EnvironmentAnalyzer
from optimizer import SectorOptimizer
from rf_model import RFModel
import os

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
        # Assume some default tilt if not present
        tilt = row.get('Tilt', 2)

        print(f"Processing {tower_name}...")
        
        # 1. Analyze environment
        p_map, context = analyzer.analyze_environment(lat, lon)
        
        # 2. Optimize sectors
        azimuths = optimizer.optimize(p_map, context)
        
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
            'Sectors_Count': len(azimuths)
        }

        for i, az in enumerate(azimuths):
            # Generate Sector Polygon
            poly_coords = rf_model.get_sector_polygon(lat, lon, az, 65, range_m)
            
            # Map Visualization
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