import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

base_dir = "./data/Annotations"

def convert_xml_to_json(xml_files):
    data = {'labels': defaultdict(list)}

    for xml_file in xml_files:
        tree = ET.parse(os.path.join(base_dir, xml_file))
        root = tree.getroot()
        filename = root.find('filename').text

        for obj in root.findall('.//object'):
            label = 1

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            data['labels'][filename].append({
                'label': label,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })

    return data

def save_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    xml_files = os.listdir(base_dir)
    json_file = './data/labels/labels_ft.json'

    converted_data = convert_xml_to_json(xml_files)
    save_json(converted_data, json_file)
