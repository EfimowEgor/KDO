import json

import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

with open('C:/Users/egore/Desktop/KR_CV/data/labels.json', 'r') as j:
    contents = json.loads(j.read())

for i, key in enumerate(contents['labels']):
    img = read_image(f'C:/Users/egore/Desktop/KR_CV/data/images/image{i}.png')
    transform = torchvision.transforms.Lambda(lambda x: x[:3])
    img = transform(img)
    boxes = []
    for val in contents['labels'][key]:
        boxes.append([val['xmin'], val['ymin'], val['xmax'], val['ymax']])

    boxes = torch.tensor(boxes, dtype=torch.int)

    img = draw_bounding_boxes(img, boxes, width=1,
                                colors="green")

    # transform this image to PIL image
    img = torchvision.transforms.ToPILImage()(img)

    # display output
    img.show()
