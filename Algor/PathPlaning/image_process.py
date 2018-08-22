import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageProcess:
    # 定义处理图函数
    def __init__(self):
        self.images_dir = os.path.join(sys.path[0], 'Images')
        if not os.path.exists(self.images_dir):
            os.mkdir(self.images_dir)
        self.data_dir = os.path.join(sys.path[0], 'Data')
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.height_water = 7759
        self.threshold_values = np.array([1500, 2500, 3500, 4650, 5500])  # 深度是多少m

    def generte_images(self):
        original_img = cv2.imread(os.path.join(self.images_dir, 'original_black.png'), 0)
        # 读取灰度数据，第二个参数表示读入的图像是灰度
        for _, value in enumerate(self.threshold_values):
            # 图像像素是从负值开始，因此需要255减去值
            _, thresh = cv2.threshold(original_img, 255 - int(value * 255 / self.height_water), 255,
                                      cv2.THRESH_BINARY_INV)
            np.save(os.path.join(self.data_dir, str(value) + 'm.npy'), thresh)
            fig = plt.figure()
            plt.imshow(thresh)
            plt.savefig(os.path.join(self.images_dir, str(value) + 'm.png'))

    def smaller(self, number=None):
        # number :int
        if number is None:
            pass
        else:
            self.threshold_values = np.array([number])
        for _, value in enumerate(self.threshold_values):
            number = value
            # 将图片变为原图大小的四分之一，取左下角
            if not os.path.exists(os.path.join(self.data_dir, str(number) + 'm.npy')):
                temp1 = self.threshold_values.copy()
                self.threshold_values = np.array([number])
                self.generte_images()
                self.threshold_values = temp1.copy()
            img = np.load(os.path.join(self.data_dir, str(number) + 'm.npy'))
            height, length = img.shape
            small_img = img[int(0.5 * height):, :int(0.5 * length)]
            small_img = cv2.resize(small_img, (int(small_img.shape[1] / 3), int(small_img.shape[0] / 3)),
                                   interpolation=cv2.INTER_AREA)
            np.save(os.path.join(self.data_dir, str(number) + 'm_small.npy'), small_img)
            fig = plt.figure()
            plt.imshow(small_img)
            plt.savefig(os.path.join(self.images_dir, str(number) + 'm_small.png'))

    def images_test(self, name):
        # 对数据进行显示
        # name = filepath(不需要加.npy)
        img = np.load(os.path.join(self.data_dir, name + '.npy'))
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    process = ImageProcess()
    process.images_test()
