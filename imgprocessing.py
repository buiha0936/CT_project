# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:52:43 2020

@author: buiha
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon
from skimage.transform import radon, rescale, resize
import scipy.io
from scipy.fftpack import fft, ifft
from Filter_types import *
import cv2
from os.path import join

# Khai báo các chương trình con
""" Áp dụng hàm lọc nhiễu Gaussian """
def average_filter(img, size = (5,5), N = 1 ):
    kernel = np.ones((5,5),np.float32)/25
    for i in range (0,N):
        img = cv2.filter2D(img,-1,kernel)
    return img

""" Áp dụng hàm lọc cho hình chiếu 2D """
def proj_filter(img, filter_name):
    img_shape = img.shape[0]
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(img, pad_width, mode='constant', constant_values=0)
    # Tạo hàm lọc trong miền tần số
    ram_filter = filter_generator(projection_size_padded,filter_name)
    # Lọc sinogram trong miền tần số
    projection = fft(img, axis=0) * ram_filter
    radon_filtered = np.real(ifft(projection, axis=0)[:img_shape, :])
    return radon_filtered

def read_numpy_3D_data(filepath):
    # Đọc dữ liệu ảnh lưu trong tệp .npy
    data = np.load(filepath)
    
    return data 

def read_raw(raw_img_path):
	# Number of Rows
	ROWS = 2944
	# Number of Columns  
	COLS = 2352
	raw_img = open(raw_img_path)  
	# Loading the input image
	print("... Load input raw image")
	img = np.fromfile(raw_img, dtype = np.uint16, count = ROWS * COLS)
	print("Dimension of the old image array: ", img.ndim)
	print("Size of the old image array: ", img.size)
	# Conversion from 1D to 2D array
	img.shape = (img.size // COLS, COLS)
	img = np.rot90(img,1)
	return img
# ------------------------------------------------------------------------------------------------ #
# ----------------------------------------- Main Program ----------------------------------------- #
# ------------------------------------------------------------------------------------------------ #

filepath = r"F:\CBCT_KC05\PIN_Projs\0.raw"
output_path = r"F:\CBCT_KC05\Image_processing\Trunhom"
 
#data = read_numpy_3D_data(filepath)

img = read_raw(filepath)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # Convert RGB sang ảnh thang xám
img = cv2.GaussianBlur(img,(15,15),0)
img[img>12500]=12500
# Logarit hình chiếu
img = np.uint16(-np.log(img/np.max(img))*15000)

cv2.normalize(img, img, 0, 65000, cv2.NORM_MINMAX)

histr = cv2.calcHist([img],[0],None,[65000],[0,65000])                    # Calculate Image Histogram

plt.subplot(221)                                  # Ảnh hình chiếu
plt.imshow(img,cmap='gray')
plt.title('Ảnh hình chiếu')
plt.subplot(222)                                  # Histogram       
plt.plot(histr)
plt.title('Histogram',fontsize=14, color='red')
plt.xlabel('Gia tri muc xam')
plt.ylabel('So diem anh')
plt.axis([0,65000, 0,5000])
plt.subplot(223)                                  # Lát cắt
plt.plot(img[int(img.shape[0]/2),:]) 
plt.title('Lát cắt ngang ảnh 2D')                             
plt.subplot(224)                                  # Lát cắt
plt.plot(img[:,int(img.shape[1]/2)])  
plt.title('Lát cắt dọc ảnh 2D')       

# recondat = []

# for i in range (0,data.shape[0]):
#     img = gaussian_filter(data[i,:,:])
#     img = -np.log(img/np.max(img))
#     cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
#     recondat.append(img)
#     filename = str(i)+'.png'
#     cv2.imwrite(join(output_path,filename),img)

# np.save(join(output_path,'recondat'),img)
"""
# img = cv2.imread('img1-2E07.kq.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

# # Thông số cấu hình hình chiếu
# SDD = 760                                # Source to Detector Distance [mm] 
# SOD = 640                                # Source to Object Distance   [mm]
# ODD = SDD - SOD                          # Object to Detector Distance [mm] 
# detector_pixel_size = 0.143              # Pixel size                  [mm] 
# detector_rows = 500                      # Vertical size of detector   [pixels].
# detector_cols = 500                      # Horizontal size of detector [pixels].
# num_of_projections = 360
# angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)
# # # Load projections.
# # print ("=> Loading projections... ")
# # projections = np.zeros((detector_rows, num_of_projections, detector_cols))
# # for i in range(num_of_projections):
# #     im = cv2.imread('Mau1.png')                # Đọc ảnh RGB
# #     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   # Convert RGB sang ảnh thang xám
# #     cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
# #     im = gaussian_filter(im)
# #     # im = -np.log(im/np.max(im))                 # Nghịch đảo màu cho giống với ảnh X quang
# #     projections[:, i, :] = im

# img = cv2.imread('Mau1.png')                # Đọc ảnh RGB
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # Convert RGB sang ảnh thang xám
# img = img[250:750, 250:750]
# # img = cv2.fastNlMeansDenoising(img)                # Hoạt động với ma trận số nguyên
# # img = -np.log(img/np.max(img))              # Convert to transmission Image
# # img = img/np.max(img)
# cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
# denoise_img = cv2.fastNlMeansDenoising(img,5)                # Hoạt động với ma trận số nguyên
# denoise_img = gaussian_filter(img)
# cv2.normalize(denoise_img, denoise_img, 0, 255, cv2.NORM_MINMAX)

# # Áp dụng hàm lọc Gaussian
# # denoise_img = gaussian_filter(img)
# # denoise_img = proj_filter(img,'cosine')

# # denoise_img = denoise_img/np.max(denoise_img)


# # Hàm lọc làm nổi cạnh trong ảnh
# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# denoise_img = cv2.filter2D(denoise_img, -1, kernel)

# #cv2.normalize(denoise_img, denoise_img, 0, 255, cv2.NORM_MINMAX)

# #edges = cv2.Canny(denoise_img,100,200)
# #CT_img = CT_img/np.max(CT_img)

# #plt.hist(denoise_img.ravel(),256,[0,256]); plt.show()
# cv2.imwrite("Mau1_crop.png", denoise_img)
# fig, axs = plt.subplots(2,2)
# axs[0,0].set_title("Lát cắt ảnh gốc")
# axs[0,0].plot(img[:,250])
# axs[0,1].set_title("Ảnh gốc")
# axs[0,1].plot(denoise_img[:,250])
# axs[1,0].set_title("Lát cắt ảnh lọc")
# axs[1,0].imshow(img, cmap=plt.cm.Greys_r)
# axs[1,1].set_title("Ảnh đã lọc")
# axs[1,1].imshow(denoise_img, cmap=plt.cm.Greys_r)
# plt.show()
"""