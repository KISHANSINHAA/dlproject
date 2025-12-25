import xml.etree.ElementTree as ET
from config import CLASSES

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    boxes = []
    class_ids = []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text) / img_w
        ymin = float(bbox.find("ymin").text) / img_h
        xmax = float(bbox.find("xmax").text) / img_w
        ymax = float(bbox.find("ymax").text) / img_h

        boxes.append([ymin, xmin, ymax, xmax])
        class_ids.append(CLASSES[label])

    return boxes, class_ids
