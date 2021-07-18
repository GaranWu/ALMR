from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import random
import os

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(img_path):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(max(pred_score))][-1]
    masks = (pred[0]['masks'] > 0.5)[0].detach().cpu().numpy()
    masks = masks[:pred_t + 1]
    return masks


def get_prediction_img(img):
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    if not pred_score:
        masks = torch.zeros_like(img)
    else:
        pred_t = [pred_score.index(max(pred_score))][-1]
        masks = (pred[0]['masks'] > 0.5)[0].detach().cpu().numpy()
        masks = masks[:pred_t + 1]
    return masks


def random_colour_masks(image):
    # colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
    # [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    colours = [255, 255, 255]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    # r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    r[image == 1], g[image == 1], b[image == 1] = colours
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def instance_segmentation_api(img_path):
    masks = get_prediction(img_path)
    rgb_mask = random_colour_masks(masks[0])
    return rgb_mask


def instance_segmentation_img(img):
    masks = get_prediction_img(img)
    rgb_mask = random_colour_masks(masks[0])
    return rgb_mask


'''
cv2.imwrite("./imgs/mask/rgb_mask.jpg", rgb_mask)
cv2.imwrite("./imgs/mask/mask.jpg", img)

plt.figure(figsize=(20, 30))
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()
'''


# instance_segmentation_api('./imgs/mask/bird.jpg')
# instance_segmentation_api('./imgs/mask/bird1.jpg')


def make_mask():
    name_list = os.listdir('../data/birds/CUB_200_2011/images')
    for file_name in name_list[8:]:
        if not os.path.exists('../data/birds/mask/' + file_name):
            os.mkdir('../data/birds/mask/' + file_name)
        for image_name in os.listdir('../data/birds/CUB_200_2011/images/' + file_name):
            mask = instance_segmentation_api('../data/birds/CUB_200_2011/images/' + file_name + '/' + image_name)
            cv2.imwrite('../data/birds/mask/' + file_name + '/' + image_name, mask)
            print('../data/birds/mask/' + file_name + '/' + image_name)


make_mask()
