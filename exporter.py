import xml.etree.ElementTree as ET
from config import parse_tech_string, get_rf_params
import numpy as np

class KMLExporter:
    def __init__(self, physics):
        self.physics = physics
        # ABGR format: Alpha Blue Green Red
        self.tech_colors = {
            'GSM': {'poly': '6600FF00', 'line': 'FF008800'},  # Transparent Green / Dark Green Border
            'UMTS': {'poly': '66FF0000', 'line': 'FF880000'}, # Transparent Blue / Dark Blue Border
            'LTE': {'poly': '660000FF', 'line': 'FF000088'},  # Transparent Red / Dark Red Border
            'NR': {'poly': '6600FFFF', 'line': 'FF008888'}    # Transparent Yellow / Dark Cyan Border
        }

    def create_kml(self, towers_df, optimized_params, output_path='network_plan.kml'):
        kml = ET.Element('kml', xmlns="http://www.opengis.net/kml/2.2")
        document = ET.SubElement(kml, 'Document')
        ET.SubElement(document, 'name').text = "JULES V01 Optimized Network Plan"

        # Styles
        for tech, colors in self.tech_colors.items():
            style = ET.SubElement(document, 'Style', id=f"style_{tech}")
            ls = ET.SubElement(style, 'LineStyle')
            ET.SubElement(ls, 'color').text = colors['line']
            ET.SubElement(ls, 'width').text = "3"
            ps = ET.SubElement(style, 'PolyStyle')
            ET.SubElement(ps, 'color').text = colors['poly']
            ET.SubElement(ps, 'fill').text = "1"
            ET.SubElement(ps, 'outline').text = "1"

        for t_idx, row in towers_df.iterrows():
            tower_folder = ET.SubElement(document, 'Folder')
            ET.SubElement(tower_folder, 'name').text = str(row['Tower_ID'])

            # Tower Placemark
            tp = ET.SubElement(tower_folder, 'Placemark')
            ET.SubElement(tp, 'name').text = f"Tower: {row['Tower_ID']}"
            point = ET.SubElement(tp, 'Point')
            ET.SubElement(point, 'coordinates').text = f"{row['Lon']},{row['Lat']},0"

            tech, freq = parse_tech_string(row['Tech_String'])
            rf_params = get_rf_params(freq)

            for s_idx in range(3):
                az = optimized_params[t_idx, s_idx, 0]
                tilt = optimized_params[t_idx, s_idx, 1]

                # Dynamic Cell Edge Calculation
                dynamic_range = self.physics.calculate_cell_edge_range(row, az, tilt, freq)

                sp = ET.SubElement(tower_folder, 'Placemark')
                ET.SubElement(sp, 'name').text = f"S{s_idx + 1} | {tech} {freq} | Az:{round(az,1)}° | R:{int(dynamic_range)}m"
                ET.SubElement(sp, 'styleUrl').text = f"#style_{tech}"

                polygon = ET.SubElement(sp, 'Polygon')
                ET.SubElement(polygon, 'tessellate').text = "1"
                outer = ET.SubElement(polygon, 'outerBoundaryIs')
                ring = ET.SubElement(outer, 'LinearRing')

                # Hard Coordinate Scaling inside get_sector_polygon
                coords = self.physics.get_sector_polygon(row['Lat'], row['Lon'], az, rf_params['hbw'], dynamic_range)
                coord_str = " ".join([f"{c[0]},{c[1]},0" for c in coords])
                ET.SubElement(ring, 'coordinates').text = coord_str

        tree = ET.ElementTree(kml)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"Professional KML exported to {output_path}")
