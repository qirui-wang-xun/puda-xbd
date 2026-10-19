import torch
import torch.utils.data as data

from PIL import Image
import os
import numpy as np


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images

class CLEFImage(data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader, pos_label_percent=5, random_state=None):
        imgs = make_dataset(root, label)
        #select random labelled positive indexes
        pos_label_index = random_state.choice(np.where([label==1 for img,label in imgs])[0], int(np.ceil(pos_label_percent/100*np.sum([label==1 for img,label in imgs]))), replace=False)
        # Describe each image as triple (feature, pu_label, label)
        imgs_pu = [(img[0],img[1], img[1]) if i in pos_label_index else (img[0],-1, img[1]) for i, img in enumerate(imgs)]
        self.root = root
        self.label = label
        self.imgs = imgs_pu
        self.transform = transform
        self.loader = loader


    def __getitem__(self, index):
        path, target_pu, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, target_pu, target, index

    def __len__(self):
        return len(self.imgs)


