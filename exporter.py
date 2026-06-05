import xml.etree.ElementTree as ET
from config import parse_tech_string, get_rf_params
import numpy as np

class KMLExporter:
    def __init__(self, physics):
        self.physics = physics
        # KML Color format is ABGR (Alpha Blue Green Red)
        # Using 66 (hex) for alpha (~40% transparency)
        self.tech_colors = {
            'GSM': {'poly': '6600FF00', 'line': 'FF008800'},  # Green transparent / Solid dark green border
            'UMTS': {'poly': '66FF0000', 'line': 'FF880000'}, # Blue transparent / Solid dark blue border
            'LTE': {'poly': '660000FF', 'line': 'FF000088'},  # Red transparent / Solid dark red border
            'NR': {'poly': '6600FFFF', 'line': 'FF008888'}    # Yellow transparent / Solid dark yellow border
        }

    def create_kml(self, towers_df, optimized_params, output_path='network_plan.kml'):
        kml = ET.Element('kml', xmlns="http://www.opengis.net/kml/2.2")
        document = ET.SubElement(kml, 'Document')
        ET.SubElement(document, 'name').text = "JULES V01 Optimized Network Plan"

        # Define Styles with Transparency and Borders
        for tech, colors in self.tech_colors.items():
            style = ET.SubElement(document, 'Style', id=f"style_{tech}")

            # LineStyle for Borders
            line_style = ET.SubElement(style, 'LineStyle')
            ET.SubElement(line_style, 'color').text = colors['line']
            ET.SubElement(line_style, 'width').text = "3"

            # PolyStyle for Transparent Fills
            poly_style = ET.SubElement(style, 'PolyStyle')
            ET.SubElement(poly_style, 'color').text = colors['poly']
            ET.SubElement(poly_style, 'fill').text = "1"
            ET.SubElement(poly_style, 'outline').text = "1"

        for t_idx, row in towers_df.iterrows():
            tower_folder = ET.SubElement(document, 'Folder')
            ET.SubElement(tower_folder, 'name').text = str(row['Tower_ID'])

            # Tower Site Location
            placemark = ET.SubElement(tower_folder, 'Placemark')
            ET.SubElement(placemark, 'name').text = f"Tower: {row['Tower_ID']}"
            point = ET.SubElement(placemark, 'Point')
            ET.SubElement(point, 'coordinates').text = f"{row['Lon']},{row['Lat']},0"

            tech, freq = parse_tech_string(row['Tech_String'])
            rf_params = get_rf_params(freq)

            for s_idx in range(3):
                az = optimized_params[t_idx, s_idx, 0]
                tilt = optimized_params[t_idx, s_idx, 1]

                # Dynamic Radius Calculation (RSRP = -95dBm bound)
                dynamic_range = self.physics.calculate_cell_edge_range(row, az, tilt, freq)

                sector_placemark = ET.SubElement(tower_folder, 'Placemark')
                ET.SubElement(sector_placemark, 'name').text = f"S{s_idx + 1} | {tech} {freq} | Az:{round(az,1)}°"
                ET.SubElement(sector_placemark, 'styleUrl').text = f"#style_{tech}"

                # Each sector is a distinct Polygon wedge
                polygon = ET.SubElement(sector_placemark, 'Polygon')
                ET.SubElement(polygon, 'tessellate').text = "1"
                outer_boundary = ET.SubElement(polygon, 'outerBoundaryIs')
                linear_ring = ET.SubElement(outer_boundary, 'LinearRing')

                # Precise arc generation with dynamic radius
                coords = self.physics.get_sector_polygon(row['Lat'], row['Lon'], az, rf_params['hbw'], dynamic_range)
                coord_str = " ".join([f"{c[0]},{c[1]},0" for c in coords])
                ET.SubElement(linear_ring, 'coordinates').text = coord_str

        tree = ET.ElementTree(kml)
        # Proper XML formatting for Google Earth
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"Professional KML exported to {output_path}")
