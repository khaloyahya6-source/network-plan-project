import xml.etree.ElementTree as ET
from config import parse_tech_string, get_tech_specs
import numpy as np

class KMLExporter:
    def __init__(self, physics):
        self.physics = physics
        # KML Color format is ABGR (Alpha Blue Green Red)
        # Using 66 (hex) for alpha (~40% transparency)
        self.tech_colors = {
            'GSM': {'poly': '6600FF00', 'line': 'FF008800'},  # Transparent Green / Dark Green Border
            'UMTS': {'poly': '66FF0000', 'line': 'FF880000'}, # Transparent Blue / Dark Blue Border
            'LTE': {'poly': '660000FF', 'line': 'FF000088'},  # Transparent Red / Dark Red Border
            'NR': {'poly': '6600FFFF', 'line': 'FF008888'}    # Transparent Yellow / Dark Cyan Border
        }

    def create_kml(self, towers_df, optimized_params, output_path='network_plan.kml'):
        kml = ET.Element('kml', xmlns="http://www.opengis.net/kml/2.2")
        doc = ET.SubElement(kml, 'Document')
        ET.SubElement(doc, 'name').text = "JULES V01 Optimized Network Plan"

        # Define Styles
        for tech, colors in self.tech_colors.items():
            style = ET.SubElement(doc, 'Style', id=f"style_{tech}")
            ls = ET.SubElement(style, 'LineStyle')
            ET.SubElement(ls, 'color').text = colors['line']
            ET.SubElement(ls, 'width').text = "3"
            ps = ET.SubElement(style, 'PolyStyle')
            ET.SubElement(ps, 'color').text = colors['poly']
            ET.SubElement(ps, 'fill').text = "1"
            ET.SubElement(ps, 'outline').text = "1"

        for t_idx, row in towers_df.iterrows():
            folder = ET.SubElement(doc, 'Folder')
            ET.SubElement(folder, 'name').text = str(row['Tower_ID'])

            # Tower Placemark
            pm = ET.SubElement(folder, 'Placemark')
            ET.SubElement(pm, 'name').text = f"Site: {row['Tower_ID']}"
            pt = ET.SubElement(pm, 'Point')
            ET.SubElement(pt, 'coordinates').text = f"{row['Lon']},{row['Lat']},0"

            tech_str = row['Tech_String']
            tech_key, freq = parse_tech_string(tech_str)
            specs = get_tech_specs(tech_str)

            for s_idx in range(3):
                az = optimized_params[t_idx, s_idx, 0]
                tilt = optimized_params[t_idx, s_idx, 1]

                # Dynamic Cell Edge radius calculation (-95 dBm bound)
                radius = self.physics.calculate_cell_edge_range(row, az, tilt, tech_str)

                spm = ET.SubElement(folder, 'Placemark')
                ET.SubElement(spm, 'name').text = f"S{s_idx+1} | {tech_key} {freq}MHz | Az:{int(az)} | R:{int(radius)}m"
                ET.SubElement(spm, 'styleUrl').text = f"#style_{tech_key}"

                poly = ET.SubElement(spm, 'Polygon')
                ET.SubElement(poly, 'tessellate').text = "1"
                outer = ET.SubElement(poly, 'outerBoundaryIs')
                ring = ET.SubElement(outer, 'LinearRing')

                # Correct Mathematical Scaling inside physics engine call
                coords = self.physics.get_sector_polygon(row['Lat'], row['Lon'], az, specs['hbw'], radius)
                c_str = " ".join([f"{c[0]},{c[1]},0" for c in coords])
                ET.SubElement(ring, 'coordinates').text = c_str

        tree = ET.ElementTree(kml)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"Professional RF KML exported: {output_path}")
