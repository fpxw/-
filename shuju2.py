import glob
import os
import cv2
from imgaug import augmenters as iaa
import imgaug as ia


all_images = glob.glob('D:/shuju/action_detection/battery/*.jpg')
#print(all_images)
for path in all_images:
    name = os.path.basename(path)[:-4]
    print(name)
    #images = cv2.imread(path,0)
    images = cv2.imread(path)
    images = [images,images,images]
    #定义一个lambda表达式，以p=0.5的概率去执行sometimes传递的图像增强
    sometimes = lambda aug:iaa.Sometimes(0.5,aug)
    #建立一个名为seq的实例，定义增强方法，用于增强
    aug =iaa.Sequential(
        [
        iaa.Fliplr(0.5),# 对50%的图像进行镜像翻转
        iaa.Flipud(0.2),#对20%的图像做左右翻转
        sometimes(iaa.Crop(percent=(0, 0.1))),
        sometimes(iaa.Affine( # 部分图像做仿射变换
            scale = {'x':(0.8,1.2),'y':(0.8,1.2)},# 图像缩放为80%到120%
            translate_percent={'x':(-0.2,0.2),'y':(-0.2,0.2)},# 平移±20%
            rotate=(-30,30),# 旋转±30度
            shear=(-16,16),# 剪切变换±16度（矩形变平行四边形）
            cval=(0,255),# 全白全黑填充
            mode=ia.ALL# 定义填充图像外区域的方法
        )),
        # 使用下面的0个到2个之间的方法增强图像
        iaa.SomeOf((0,2),
           [
            iaa.Sharpen(alpha=(0,0.3),lightness=(0.9,1.1)),#锐化处理
            # 加入高斯噪声
            iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.05*255),per_channel=0.5),
            iaa.Add((-10,10),per_channel=0.5),# 每个像素随机加减-10到10之间的数
            iaa.Multiply((0.8,1.2),per_channel=0.5),# 像素乘上0.5或者1.5之间的数字
            # 将整个图像的对比度变为原来的一半或者二倍
            iaa.ContrastNormalization((0.5,2.0),per_channel=0.5),
            ],
           random_order=True)
        ],
    random_order=True # 随机的顺序把这些操作用在图像上
    )

    images_aug = aug.augment_images(images)# 应用数据增强
    n = 0
    for each in images_aug:
        #保存到指定路径
        #cv2.imwrite('D:/shuju/action_detection/new_jpg/%s%s.jpg'%(name,n),each)
        cv2.imwrite('D:/shuju/action_detection/new_jpg/%s.jpg'%(name),each)
        n += 1