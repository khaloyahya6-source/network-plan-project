import pandas as pd
import numpy as np
import folium
import re
from analyzer import EnvironmentAnalyzer
from optimizer import SectorOptimizer
from rf_model import RFModel
from shapely.geometry import Polygon

def fuzzy_parse_bands(band_string):
    """
    Map strings like 'U900', 'L1800', 'n78', 'G9' to Low/Mid/High layers.
    """
    if not isinstance(band_string, str): return []

    bands = []
    # Split by common delimiters
    tokens = re.split(r'[,|/; ]+', band_string)

    for t in tokens:
        t = t.upper().strip()
        # Regex to find numbers
        match = re.search(r'(\d+)', t)
        if not match: continue

        freq = int(match.group(1))
        # Handle cases like '9' for 900
        if freq < 100: freq *= 100

        if freq <= 1000:
            bands.append(('Low', t, freq))
        elif freq <= 2200:
            bands.append(('Mid', t, freq))
        else:
            bands.append(('High', t, freq))
    return bands

def main():
    print("Initializing RF Planning Tool...")

    try:
        # Load Sites
        df_sites = pd.read_excel('sites.xlsx', sheet_name='sites')
        # Load Configs (Fuzzy Band Matching)
        df_config = pd.read_excel('sites.xlsx', sheet_name='configuration')
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return

    analyzer = EnvironmentAnalyzer()
    optimizer = SectorOptimizer()
    rf_model = RFModel()

    global_coverage_polygons = []
    results = []

    # Map Setup
    m = folium.Map(location=[df_sites['Latitude'].mean(), df_sites['Longitude'].mean()], zoom_start=13)
    folium.TileLayer('cartodbpositron', name='Street Map').add_to(m)

    for idx, row in df_sites.iterrows():
        site_id = row['Site ID']
        site_name = row['Site Name']
        lat, lon = row['Latitude'], row['Longitude']
        total_height = row['Total Height']

        print(f"Processing Site: {site_id} ({site_name})...")

        # 1. Get Bands for this site
        site_config = df_config[df_config['Site ID'] == site_id]
        if site_config.empty:
            print(f"No config found for {site_id}, skipping.")
            continue

        raw_bands = site_config.iloc[0]['Bands']
        parsed_bands = fuzzy_parse_bands(raw_bands)

        # 2. Analyze Environment
        osm_data = analyzer.fetch_osm_data(lat, lon)
        site_type = analyzer.classify_site(lat, lon, osm_data)

        # 3. Optimize Azimuths
        azimuths = optimizer.optimize_azimuths(lat, lon, site_type, analyzer, rf_model, parsed_bands, osm_data, global_coverage_polygons)

        # 4. Generate Sectors & Map Layers
        for s_idx, az in enumerate(azimuths):
            sector_id = s_idx + 1

            # For each band, calculate range and draw concentric polygons
            # Sort bands: Low (largest) first to High (smallest) so they stack correctly on map
            sorted_bands = sorted(parsed_bands, key=lambda x: x[2])

            # We calculate clutter score based on the Mid band range as a representative sample
            sample_range = rf_model.calculate_range('Mid', 0.5, total_height)
            clutter = analyzer.get_sector_clutter_score(lat, lon, az, 65, sample_range, osm_data)

            for band_type, band_name, freq in sorted_bands:
                calc_range = rf_model.calculate_range(band_type, clutter, total_height)
                poly_coords = rf_model.get_sector_polygon(lat, lon, az, 65, calc_range)

                # Add to global coverage (using Low band only for whitespace filling)
                if band_type == 'Low':
                    global_coverage_polygons.append(Polygon([(p[1], p[0]) for p in poly_coords]))

                # Map Colors
                color_map = {'Low': 'green', 'Mid': 'blue', 'High': 'red'}

                # Tooltip logic
                tooltip_html = f"Site:{site_id}, S:{sector_id}<br><br>INFORMATION<br>Site ID: {site_id}<br>Site Name: {site_name}<br>SectorID: {sector_id}<br>Azimuth: {az}<br>Beamwidth: 65<br>GPS Lat: {lat}<br>GPS Lon: {lon}<br><br>DATA<br>Layer 1 Height (m): {total_height}"

                folium.Polygon(
                    locations=poly_coords,
                    color=color_map.get(band_type, 'gray'),
                    fill=True,
                    fill_opacity=0.2,
                    weight=1,
                    tooltip=tooltip_html
                ).add_to(m)

            results.append({
                'Site ID': site_id,
                'Site Name': site_name,
                'Sector': sector_id,
                'Azimuth': az,
                'Type': site_type,
                'Bands': raw_bands,
                'Clutter': round(clutter, 2)
            })

    # Save Output
    pd.DataFrame(results).to_excel('RF_Planning_Results.xlsx', index=False)
    m.save('RF_Coverage_Map.html')
    print("Planning Complete. Files generated: RF_Planning_Results.xlsx, RF_Coverage_Map.html")

if __name__ == "__main__":
    main()
