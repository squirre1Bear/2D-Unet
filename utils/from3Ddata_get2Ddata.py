import numpy as np

'''
有效集
'''
image_list = np.load(r"E:\LIDC_dataset\center_data\valid_center_image_list.npy")
mask_list = np.load(r"E:\LIDC_dataset\center_data\valid_center_label_list.npy")

print(image_list.shape)

depth_index = 12
# 记录总样本数
num_samples = image_list.shape[0]
# 得到每一个3D数据的中心图像
image_2D_list = image_list[:, depth_index, :, :]
mask_2D_list = mask_list[:, depth_index, :, :]

np.save(r'E:\LIDC_dataset\2D_center_data\valid_center_image_list.npy', image_2D_list)
np.save(r'E:\LIDC_dataset\2D_center_data\valid_center_label_list.npy', mask_2D_list)

'''
测试集
'''
# image_list = np.load(r"E:\LIDC_dataset\center_data\test_center_image_list.npy")
# mask_list = np.load(r"E:\LIDC_dataset\center_data\test_center_label_list.npy")
#
# print(image_list.shape)
#
# depth_index = 12
# # 记录总样本数
# num_samples = image_list.shape[0]
# # 得到每一个3D数据的中心图像
# image_2D_list = image_list[:, depth_index, :, :]
# mask_2D_list = mask_list[:, depth_index, :, :]
#
# np.save(r'E:\LIDC_dataset\2D_center_data\test_center_image_list.npy', image_2D_list)
# np.save(r'E:\LIDC_dataset\2D_center_data\test_center_label_list.npy', mask_2D_list)


'''
训练集
'''
# image_list = np.load(r"E:\LIDC_dataset\center_data\train_center_image_list.npy")
# mask_list = np.load(r"E:\LIDC_dataset\center_data\train_center_label_list.npy")
#
# print(image_list.shape)
#
# depth_index = 12
# # 记录总样本数
# num_samples = image_list.shape[0]
# # 得到每一个3D数据的中心图像
# image_2D_list = image_list[:, depth_index, :, :]
# mask_2D_list = mask_list[:, depth_index, :, :]
#
# np.save(r'E:\LIDC_dataset\2D_center_data\train_center_image_list.npy', image_2D_list)
# np.save(r'E:\LIDC_dataset\2D_center_data\train_center_label_list.npy', mask_2D_list)