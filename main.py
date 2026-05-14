import pandas as pd
import numpy as np
import folium
import re
from analyzer import EnvironmentAnalyzer
from optimizer import SectorOptimizer
from rf_model import RFModel
from shapely.geometry import Polygon

def fuzzy_parse_bands(band_string):
    """Parses inconsistent band strings into Low, Mid, and High layers."""
    if not isinstance(band_string, str): return []
    bands = []
    for t in re.split(r'[,|/; ]+', band_string):
        match = re.search(r'(\d+)', t.upper().strip())
        if not match: continue
        freq = int(match.group(1))
        if freq < 100: freq *= 100
        layer = 'Low' if freq <= 1000 else 'Mid' if freq <= 2200 else 'High'
        bands.append((layer, t.upper().strip(), freq))
    return bands

def main():
    print("Initializing Automated RF Planning Tool...")
    try:
        # Single-sheet input structure
        df_sites = pd.read_excel('sites.xlsx')
    except Exception as e:
        print(f"Excel Error: {e}"); return

    analyzer, optimizer, rf_model = EnvironmentAnalyzer(), SectorOptimizer(), RFModel()
    global_polys, results = [], []
    m = folium.Map(location=[df_sites['Latitude'].mean(), df_sites['Longitude'].mean()], zoom_start=13)

    for _, row in df_sites.iterrows():
        s_id, s_name, lat, lon, h = row['Site ID'], row['Site Name'], row['Latitude'], row['Longitude'], row['Total Height']
        raw_bands = str(row.get('Bands', ''))
        parsed_bands = fuzzy_parse_bands(raw_bands)
        if not parsed_bands: continue

        print(f"Processing Site: {s_id}...")
        osm_data = analyzer.fetch_osm_data(lat, lon)
        site_type = analyzer.classify_site(lat, lon, osm_data)
        azimuths = optimizer.optimize_azimuths(lat, lon, site_type, analyzer, rf_model, osm_data, global_polys)

        for s_idx, az in enumerate(azimuths):
            # Use Mid band range for clutter sampling
            sample_r = rf_model.calculate_range('Mid', 0.5, h)
            clutter = analyzer.get_sector_clutter_score(lat, lon, az, 65, sample_r, osm_data)

            # Stack concentric polygons (Low at bottom)
            for b_layer, b_name, freq in sorted(parsed_bands, key=lambda x: x[2]):
                calc_r = rf_model.calculate_range(b_layer, clutter, h)
                coords = rf_model.get_sector_polygon(lat, lon, az, 65, calc_r)
                if b_layer == 'Low': global_polys.append(Polygon([(p[1], p[0]) for p in coords]))

                # Requested Tooltip Structure
                tooltip = f"Site:{s_id}, S:{s_idx+1}<br><br>INFORMATION<br>Site ID: {s_id}<br>Site Name: {s_name}<br>SectorID: {s_idx+1}<br>Azimuth: {az}<br>Beamwidth: 65<br>GPS Lat: {lat}<br>GPS Lon: {lon}<br><br>DATA<br>Layer 1 Height (m): {h}"

                folium.Polygon(
                    locations=coords,
                    color={'Low':'green','Mid':'blue','High':'red'}.get(b_layer,'gray'),
                    fill=True, fill_opacity=0.3, weight=1, tooltip=tooltip
                ).add_to(m)

            results.append({
                'Site ID': s_id, 'Site Name': s_name, 'Sector': s_idx+1,
                'Azimuth': az, 'Type': site_type, 'Clutter': round(clutter, 2)
            })

    # Save final reports
    pd.DataFrame(results).to_excel('RF_Planning_Results.xlsx', index=False)
    m.save('RF_Coverage_Map.html')
    print("Done. Generated RF_Planning_Results.xlsx and RF_Coverage_Map.html")

if __name__ == "__main__":
    main()
