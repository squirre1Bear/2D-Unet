import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import json
from unet2D import UNet2D
from utils import readDataList
from train import MyTransform

with open(r'train_config.json', encoding="utf-8") as f:
    config = json.load(f)

# 设定设备
device = torch.device('cuda:0')

# 加载模型
model = UNet2D(in_channels=1, num_classes=config['num_classes'])
model.load_state_dict(torch.load(config['trained_model_path'], weights_only=True))
model.eval()
model.to(device)

# 数据预处理
transform = MyTransform()

# 读取测试数据集
test_dataset = readDataList.UnetDataset(transform=transform, images_dir=config['testPic_dir'], labels_dir=config['testLab_dir'], mode='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 可视化预测结果
def visualize_prediction(data, target, output, ii=1):
    with open(r'train_config.json', encoding="utf-8") as f:
        config = json.load(f)
    save_path = config["predict_img_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image = data.cpu().numpy()  # 提取原影像。array, 32*56
    label = target.cpu().numpy()  # 提取真实掩码
    pred = output.detach().cpu().numpy().argmax(axis=0)  # 获取预测类别，选择最大概率的类别
    pred[pred==1] = 255

    # 下面这段是保存对比图（原图+正确标签+预测标签）的结果
    # 绘图
    '''
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(label, cmap='gray')
    axes[1].set_title('True Labels')
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Predictions')

    '''
    # 下面这段只保存预测标签
    # 先使用imshow将pred转成图片，之后用plt.save即可保存
    plt.imshow(pred, cmap='gray')
    plt.axis('off')


    plt.savefig(os.path.join(save_path, f'{ii}_Unet.png'))
#    plt.show()
    plt.close()
    print("图片"+str(ii) + "录入完成")

# 遍历测试集进行预测
with torch.no_grad():
    ii = 800
    # 初始化基本变量
    loss_sum = 0.0  # 总损失值
    correct_sum = 0.0  # 准确个数
    labeled_sum = 0.0
    inter_sum = 0.0
    union_sum = 0.0
    pixelAcc = 0.0
    IoU = 0.0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)    # tensor, 1*1*32*56
        # 模型预测#
        output = model(data)    # tensor, 1*2*32*56

        # 计算指标
        oriPic = np.squeeze(data, axis=0).cpu().numpy()
        prediction = np.squeeze(output, axis=0).cpu().numpy()  # 2*32*56，显示结果正常
        oriLabel = np.squeeze(target, axis=0).cpu().numpy()

        # 下面开始计算IoU
        prediction = np.argmax(prediction, axis=0)
        prediction = (prediction + 1).astype(np.int64)
        oriLabel = (oriLabel + 1).astype(np.int64)

        # 计算标注了的像素数
        pixel_labeled = (oriLabel > 0).sum()

        # 计算预测正确的像素数
        pixel_correct = ((prediction == oriLabel) * (oriLabel > 0)).sum()  # 预测正确的像素数

        # 计算交集
        prediction = (prediction * (oriLabel > 0)).astype(np.int64)
        intersection = prediction * (prediction == oriLabel).astype(np.int64)

        # 计算每个类别的交集、预测和标签的像素数量
        area_inter = np.histogram(intersection, bins=2, range=(1, 3))[0]
        area_pred = np.histogram(prediction, bins=2, range=(1, 3))[0]
        area_lab = np.histogram(oriLabel, bins=2, range=(1, 3))[0]
        # 计算并集
        area_union = area_pred + area_lab - area_inter

        # np.round:四舍五入
        correct = np.round(pixel_correct, 5)
        labeled = np.round(pixel_labeled, 5)
        inter = np.round(area_inter, 5)
        union = np.round(area_union, 5)

        correct_sum += correct
        labeled_sum += labeled
        inter_sum += inter
        union_sum += union

        # np.spaceing(1)是一个非常小的整数，用来防止“/0”的出现
        pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)  # 预测正确的像素数 / 预测了的所有像素
        IoU = (1.0 * inter_sum / (np.spacing(1) + union_sum)).mean()  # 每个类别都求交集数/并集数
        print("图片{}：Acc={:.2f} | mIoU={:.4f}".format(ii, pixelAcc, IoU))

        # 可视化结果。这里的“[0]”是舍弃了第一个维度（inchannel_num）
        visualize_prediction(data[0][0], target[0][0], output[0], str(ii))
        ii = ii + 1

print("预测结果已保存！")