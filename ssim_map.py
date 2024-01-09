import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def cal_ssim(img1, img2):
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T

    # M, N = np.shape(img1)

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    img1 = np.float64(img1)
    img2 = np.float64(img2)

    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = signal.convolve2d(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2 * img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim, ssim_map


# # import os
# # import matplotlib.pyplot as plt
# # import scipy.misc
# #
# # output_directory = os.path.dirname('data/data_128/new_g/')  # 提取文件的路径
# # output_name = os.path.splitext(os.path.basename("feature2.npy"))[0]  # 提取文件名
# # arr = np.load('data/data_128/new_g/feature2.npy')  # 提取 npy 文件中的数组
# # # disp_to_img = scipy.misc.imresize( arr , [128, 128])  # 根据 需要的尺寸进行修改
# # plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), arr, cmap=plt.cm.seismic)  # 定义命名规则，保存图片
# #
#
# img1 = np.load('data/data_128/new_g/feature2.npy')
# img2=img1
# print(img2.shape)
# # # Assuming single channel images are read. For RGB image, uncomment the following commented lines
# # img1 = cv2.imread('images/feature1_disp.png', 0)
# # # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# # img2 = cv2.imread('images/feature2_disp.png', 0)
# # # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
# ssim_index, ssim_map = cal_ssim(img1, img2)
# print(ssim_index)
# print(ssim_map.shape)
# plt.imshow(ssim_map, cmap=plt.cm.seismic, interpolation='bilinear')
# plt.title("Feature")
# plt.show()