import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from skimage.measure import find_contours
import cv2
from PIL import Image
import pylidc as pl
from pylidc.utils import consensus
import os
import torch

# 输入文件夹路径，得到读取后的图像、掩膜张量。
def get_tensor(dataset_path):
    ii = 0
    # 大小都是8的倍数，是为了最大池化的时候尺寸大小不会出现小数，防止特征图大小不同无法合并。

    # 设置分块的形状大小，24*32*56的大小差不多能涵盖80%以上的结节。
    nod_d = 24
    nod_h = 32
    nod_w = 56

    image_list = []
    cmask_list = []
    wrong_dicom = []
    # Query for a scan, and convert it to an array volume.
    for dicom_name in os.listdir(dataset_path):
        print(dicom_name)
        PathDicom = os.path.join(dicom_name)  # 获得每个Dicom文件的路径
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == dicom_name).first()  # 按名字读取Dicom文件
        vol = scan.to_volume()  # 转换成3D数组,512*512*Z，Z的大小不定

        # Cluster the annotations for the scan, and grab one.
        nods = scan.cluster_annotations()  # 获取聚类后的annotation，长度等于病人结节数量。

        try:  # 防止有的文件没有annotation
            anns = nods[0]
            Malignancy = anns[0]
            Malignancy = Malignancy.Malignancy
        except:
            pass
        # 有50%共识的时候就认为是结节
        # We pad the slices to add context for viewing.
        cmask, cbbox, masks = consensus(anns, clevel=0.5,
                                        pad=[(0, 0), (7, 7), (20, 20)])

        # 下面的image是原始图像，数组！cmask是对应的掩码(true/false)！
        image = vol[cbbox]  # 用cbbox定义的边界框，从体积数据vol中提取切片。
        image = normalize_hu(image)

        cmask = cmask[image.shape[0] // 2]
        image = image[image.shape[0] // 2]

        # 下面开始按分块大小进行图形切割、填充
        if cbbox[1].stop-cbbox[1].start > nod_h:
            image = image[image.shape[0] // 2 - nod_h // 2: image.shape[0] // 2 + nod_h // 2, :]
            cmask = cmask[cmask.shape[0] // 2 - nod_h // 2: cmask.shape[0] // 2 + nod_h // 2, :]
        if cbbox[2].stop-cbbox[2].start > nod_w:
            image = image[:, image.shape[1] // 2 - nod_w // 2: image.shape[1] // 2 + nod_w // 2]
            cmask = cmask[:, cmask.shape[1] // 2 - nod_w // 2: cmask.shape[1] // 2 + nod_w // 2]

        pad = (
            (max(0, (nod_h - image.shape[0]) // 2), max(0, nod_h - image.shape[0] - (nod_h - image.shape[0]) // 2)),
            (max(0, (nod_w - image.shape[1]) // 2), max(0, nod_w - image.shape[1] - (nod_w - image.shape[1]) // 2)),
        )
        # 填充图像
        image = np.pad(image, pad, mode='constant', constant_values=0)
        cmask = np.pad(cmask, pad, mode='constant', constant_values=0)

        print(image.shape)
        print(cmask.shape)

        if image.shape != (nod_h, nod_w) or cmask.shape != (nod_h, nod_w):
            print("有个形状错误的数据" + dicom_name)
            wrong_dicom.append(dicom_name)
            wrong_dicom.append(image.shape)
            wrong_dicom.append(cmask.shape)
            continue

        image_list.append(image)
        cmask_list.append(cmask)

        ii += 1

        # plt.imshow(image, cmap='gray')
        # plt.axis('off')
        # plt.savefig(os.path.join(r"E:\LIDC_dataset\2D_center_data\train_pic", f'trainPic_{ii-1}.png'),bbox_inches='tight')
        # plt.close()

        # 将图像转换为 uint8 类型
        image = image * 255
        cmask = cmask * 255
        image = image.astype(np.uint8)
        cmask = cmask.astype(np.uint8)
        # 将图像转换为灰度模式
        im = Image.fromarray(image).convert('L')
        cm = Image.fromarray(cmask).convert('L')
        im.save(os.path.join(r"E:\LIDC_dataset\2D_center_data\test_pic", f'testPic_{ii+799}.png'))
        cm.save(os.path.join(r"E:\LIDC_dataset\2D_center_data\test_label", f'testLabel_{ii+799}.png'))

    # 返回是list，里面存的数据是相同大小（32*56）的图片image
    return image_list, cmask_list, wrong_dicom

# 归一化
def normalize_hu(image):  # 图像是int16类型，总共49个通道，和z的范围大小一样。
    # 将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

# image_list, cmask_list, wrong_dicom = get_tensor(dataset_path=r"E:\LIDC_dataset\valid_dataset")
# np.save(r'E:\LIDC_dataset\2D_center_data\valid_center_image_list.npy', image_list)
# np.save(r'E:\LIDC_dataset\2D_center_data\valid_center_label_list.npy', cmask_list)
# print("数据大小有误被剔除的文件：")
# for dicom_name in wrong_dicom:
#     print(dicom_name)

# image_list, cmask_list, wrong_dicom = get_tensor(dataset_path=r"E:\LIDC_dataset\valid_dataset")
# np.save(r'E:\LIDC_dataset\2D_center_data\valid_center_image_list.npy', image_list)
# np.save(r'E:\LIDC_dataset\2D_center_data\valid_center_label_list.npy', cmask_list)
# print("数据大小有误被剔除的文件：")
# for dicom_name in wrong_dicom:
#     print(dicom_name)

image_list, cmask_list, wrong_dicom = get_tensor(dataset_path=r"E:\LIDC_dataset\test_dataset")
np.save(r'E:\LIDC_dataset\2D_center_data\test_center_image_list.npy', image_list)
np.save(r'E:\LIDC_dataset\2D_center_data\test_center_label_list.npy', cmask_list)
print("数据大小有误被剔除的文件：")
for dicom_name in wrong_dicom:
    print(dicom_name)