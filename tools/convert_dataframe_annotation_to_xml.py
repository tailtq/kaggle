import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString


def list_files_ignore_ds_store(path):
    return [file for file in os.listdir(path) if file != '.DS_Store']


def generate_xml(training_path, image_name, extension, group=None):
    image_path = '{}/{}.{}'.format(training_path, image_name, extension)

    image = cv2.imread(image_path)
    (height, width, channels) = image.shape

    # ------------- XML -------------
    root = Element('annotation')

    folder = SubElement(root, 'folder')
    folder.text = 'VOC2012'

    filename = SubElement(root, 'filename')
    filename.text = '{}.jpg'.format(image_name)

    path = SubElement(root, 'path')
    path.text = '{}/{}.jpg'.format(training_path, image_name)

    source = SubElement(root, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'

    size = SubElement(root, 'size')
    size_element = SubElement(size, 'width')
    size_element.text = str(width)

    size_element = SubElement(size, 'height')
    size_element.text = str(height)

    size_element = SubElement(size, 'depth')
    size_element.text = str(channels)

    segment_element = SubElement(root, 'segmented')
    segment_element.text = '0'

    if group is not None:
        for (index, element) in group.iterrows():
            object_element = SubElement(root, 'object')

            name = SubElement(object_element, 'name')
            name.text = 'st'

            pose = SubElement(object_element, 'pose')
            pose.text = 'Unspecified'

            truncated = SubElement(object_element, 'truncated')
            truncated.text = '0'

            difficult = SubElement(object_element, 'difficult')
            difficult.text = '0'

            # location
            bndbox = SubElement(object_element, 'bndbox')

            x_min = SubElement(bndbox, 'xmin')
            x_min.text = str(element['x_min'])

            y_min = SubElement(bndbox, 'ymin')
            y_min.text = str(element['y_min'])

            x_max = SubElement(bndbox, 'xmax')
            x_max.text = str(element['x_max'])

            y_max = SubElement(bndbox, 'ymax')
            y_max.text = str(element['y_max'])

    return root


parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Project path along with dataset[train, annotations] directories')

args = parser.parse_args()
relative_path = args.path

dataset_path = '{}/{}/dataset'.format(os.getcwd(), relative_path)
inner_path = '{}/VOC2012'.format(dataset_path)
training_path = '{}/ImageSets'.format(inner_path)
annotation_path = '{}/Annotations'.format(inner_path)
extension = 'jpg'

# list dataset and change format
df = pd.read_csv('{}/train.csv'.format(dataset_path))
df.head()

bboxes = np.array(df.apply(lambda x: json.loads(x['bbox']), axis=1).tolist()).astype(int)

df['x_min'] = bboxes[:, 0]
df['y_min'] = bboxes[:, 1]
df['x_max'] = bboxes[:, 0] + bboxes[:, 2]
df['y_max'] = bboxes[:, 1] + bboxes[:, 3]

groups = df.groupby('image_id')

# list missing images
images = list_files_ignore_ds_store(training_path)
missing_images = []

keys = groups.groups.keys()

for image in images:
    image_name = image.split('.')[0]
    if image_name not in keys:
        missing_images.append(image_name)

# generate labels for images
for image_name, group in groups:
    root = generate_xml(training_path, image_name, extension, group)
    f = open('{}/{}.xml'.format(annotation_path, image_name), 'w')
    f.write(parseString(tostring(root, encoding='unicode')).toprettyxml(indent='    '))
    f.close()

# generate labels for missing images
for image in missing_images:
    root = generate_xml(training_path, image, extension)
    f = open('{}/{}.xml'.format(annotation_path, image), 'w')
    f.write(parseString(tostring(root, encoding='unicode')).toprettyxml(indent='    '))
    f.close()
