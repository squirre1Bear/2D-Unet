import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import os

# 使用torch.utils.dataset将图像转换为张量。
# 使用之前需要重写里面 init getitem len方法。
class UnetDataset(Dataset):
    # 传入的是.png格式的文件，这里读取的时候是传入文件夹路径，使用os.listdir获得所有文件名称。之后在__getitem__时通过遍历文件名称获得对应图片
    def __init__(self, transform, images_dir, labels_dir, mode='train'):
        super().__init__()
        # self.images = np.load(images_path)
        # self.labels = np.load(labels_path)
        self.mode = mode
        self.transform = transform
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.images_name = sorted(os.listdir(images_dir))
        self.labels_nane = sorted(os.listdir(labels_dir))

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.images_name[index])
        label_path = os.path.join(self.labels_dir, self.labels_nane[index])
        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')
        # image = self.images[index]
        # label = self.labels[index]

        image, label = self.transform(image, label, self.mode)

        # fig, axis = plt.subplots(1, 2)
        # axis[0].imshow(image[0], cmap='gray')
        # axis[1].imshow(label[0], cmap='gray')
        # plt.show()

        image = image.float().cuda()
        label = label.long().cuda()

        return image, label

    def __len__(self):
        return len(self.images_name)