#coding:utf-8
from imgaug import augmenters as iaa
import numpy as np
import imgaug as ia
import cv2
images = np.array(
	[ia.quokka(size=(256, 256)) for _ in range(32)],
	dtype=np.uint8
)######同一张复制32张，模拟网络输入的batch大小 shape（32,64,64,3）
#########使用该库，输入要求4D张量（N, height, width, channels） 或者3D张量RGB类型（height, width, channels）
########## gray类型（height, width, 1）
images = images[0]
#cv2.imwrite('battery/img1.jpg',images)
images = np.expand_dims(images,axis=0)
seq = iaa.Sequential([
	iaa.Fliplr(p=0.5), # 水平0.5的概率翻转
	iaa.Flipud(p=1), ##垂直0.5的概率翻转
	iaa.Crop(percent=(0, 0.3),keep_size=True), # 0-0.1的数值，分别乘以图片的宽和高为剪裁的像素个数，保持原尺寸
	iaa.Sometimes(p=1,
					then_list=[iaa.GaussianBlur(sigma=(0, 0.5))],else_list=[iaa.Flipud(p=0.5),iaa.Flipud(p=0.5)]
	),       ######以p的概率执行then_list的增强方法，以1-p的概率执行else_list的增强方法，其中then_list,else_list默认为None
	iaa.ContrastNormalization((0.75, 1.5),per_channel=True),####0.75-1.5随机数值为alpha，对图像进行对比度增强，该alpha应用于每个通道
	iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.3*255), per_channel=0.5), #### loc 噪声均值，scale噪声方差，50%的概率，对图片进行添加白噪声并应用于每个通道
	iaa.Multiply((0.8, 1.2), per_channel=0.2), ####20%的图片像素值乘以0.8-1.2中间的数值,用以增加图片明亮度或改变颜色
	iaa.Affine(
		scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
		translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
		rotate=(-25, 25),
		shear=(-8, 8))
	########对图片进行仿射变化，缩放x,y取值范围均为0.8-1.2之间随机值，左右上下移动的值为-0.2-0.2乘以宽高后的随机值
	########旋转角度为-25到25的随机值，shear剪切取值范围0-360，-8到8随机值进行图片剪切
	], random_order=True) # 打乱定义图像增强的顺序

images_aug = seq.augment_images(images)
images_aug = np.squeeze(images_aug,axis=0)
cv2.imwrite('new_jpg/all.jpg',images_aug)