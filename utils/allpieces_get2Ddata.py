import numpy as np

'''
有效集
'''
image_list = np.load(r"E:\LIDC_dataset\center_data\valid_center_image_list.npy")
mask_list = np.load(r"E:\LIDC_dataset\center_data\valid_center_label_list.npy")

print(image_list.shape)
image_2D_list = image_list.reshape(image_list.shape[0]*image_list.shape[1], image_list.shape[2], image_list.shape[3])
mask_2D_list = mask_list.reshape(mask_list.shape[0]*mask_list.shape[1], mask_list.shape[2], mask_list.shape[3])

np.save(r'E:\LIDC_dataset\2D_center_data\allpieces_valid_center_image_list.npy', image_2D_list)
np.save(r'E:\LIDC_dataset\2D_center_data\allpieces_valid_center_label_list.npy', mask_2D_list)

'''
测试集
'''
# image_list = np.load(r"E:\LIDC_dataset\center_data\test_center_image_list.npy")
# mask_list = np.load(r"E:\LIDC_dataset\center_data\test_center_label_list.npy")
#
# print(image_list.shape)
# image_2D_list = image_list.reshape(image_list.shape[0]*image_list.shape[1], image_list.shape[2], image_list.shape[3])
# mask_2D_list = mask_list.reshape(mask_list.shape[0]*mask_list.shape[1], mask_list.shape[2], mask_list.shape[3])

# np.save(r'E:\LIDC_dataset\2D_center_data\allpieces_test_center_image_list.npy', image_2D_list)
# np.save(r'E:\LIDC_dataset\2D_center_data\allpieces_test_center_label_list.npy', mask_2D_list)


'''
训练集
'''
# image_list = np.load(r"E:\LIDC_dataset\center_data\train_center_image_list.npy")
# mask_list = np.load(r"E:\LIDC_dataset\center_data\train_center_label_list.npy")
#
# print(image_list.shape)
# image_2D_list = image_list.reshape(image_list.shape[0]*image_list.shape[1], image_list.shape[2], image_list.shape[3])
# mask_2D_list = mask_list.reshape(mask_list.shape[0]*mask_list.shape[1], mask_list.shape[2], mask_list.shape[3])
#
# np.save(r'E:\LIDC_dataset\2D_center_data\allpieces_train_center_image_list.npy', image_2D_list)
# np.save(r'E:\LIDC_dataset\2D_center_data\allpieces_train_center_label_list.npy', mask_2D_list)