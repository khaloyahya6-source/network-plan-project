import xml.etree.ElementTree as ET
from config import parse_tech_string, get_rf_params
import numpy as np

class KMLExporter:
    def __init__(self, physics):
        self.physics = physics
        self.colors = {
            'GSM': 'ff00ff00',  # Green
            'UMTS': 'ffff0000', # Blue
            'LTE': 'ff0000ff',  # Red
            'NR': 'ff00ffff'    # Yellow
        }

    def create_kml(self, towers_df, optimized_params, output_path='network_plan.kml'):
        kml = ET.Element('kml', xmlns="http://www.opengis.net/kml/2.2")
        document = ET.SubElement(kml, 'Document')
        ET.SubElement(document, 'name').text = "JULES V01 Optimized Network Plan"

        # Styles
        for tech, color in self.colors.items():
            style = ET.SubElement(document, 'Style', id=f"style_{tech}")
            poly_style = ET.SubElement(style, 'PolyStyle')
            ET.SubElement(poly_style, 'color').text = color
            ET.SubElement(poly_style, 'fill').text = "1"
            ET.SubElement(poly_style, 'outline').text = "1"
            line_style = ET.SubElement(style, 'LineStyle')
            ET.SubElement(line_style, 'color').text = color
            ET.SubElement(line_style, 'width').text = "2"

        for t_idx, row in towers_df.iterrows():
            tower_folder = ET.SubElement(document, 'Folder')
            ET.SubElement(tower_folder, 'name').text = str(row['Tower_ID'])

            # Tower Placemark
            placemark = ET.SubElement(tower_folder, 'Placemark')
            ET.SubElement(placemark, 'name').text = f"Tower: {row['Tower_ID']}"
            point = ET.SubElement(placemark, 'Point')
            ET.SubElement(point, 'coordinates').text = f"{row['Lon']},{row['Lat']},0"

            tech, freq = parse_tech_string(row['Tech_String'])
            rf_params = get_rf_params(freq)

            # Simple Range Calculation for KML (based on tech/freq)
            base_range = 1000 if freq < 1000 else 500 if freq < 3000 else 300

            for s_idx in range(3):
                az = optimized_params[t_idx, s_idx, 0]

                sector_placemark = ET.SubElement(tower_folder, 'Placemark')
                ET.SubElement(sector_placemark, 'name').text = f"Sector {s_idx + 1} ({tech} {freq}MHz)"
                ET.SubElement(sector_placemark, 'styleUrl').text = f"#style_{tech}"

                polygon = ET.SubElement(sector_placemark, 'Polygon')
                ET.SubElement(polygon, 'tessellate').text = "1"
                outer_boundary = ET.SubElement(polygon, 'outerBoundaryIs')
                linear_ring = ET.SubElement(outer_boundary, 'LinearRing')

                # Get Sector Polygon Coords
                coords = self.physics.get_sector_polygon(row['Lat'], row['Lon'], az, rf_params['hbw'], base_range)
                coord_str = " ".join([f"{c[0]},{c[1]},0" for c in coords])
                ET.SubElement(linear_ring, 'coordinates').text = coord_str

        tree = ET.ElementTree(kml)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"KML exported to {output_path}")
