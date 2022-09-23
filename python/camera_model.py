################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import re
import os
import numpy as np
import scipy.interpolate as interp
from scipy.ndimage import map_coordinates


class CameraModel:
    """Provides intrinsic parameters and undistortion LUT for a camera.

    Attributes:
        camera (str): Name of the camera.
        camera sensor (str): Name of the sensor on the camera for multi-sensor cameras.
        focal_length (tuple[float]): Focal length of the camera in horizontal and vertical axis, in pixels.
        principal_point (tuple[float]): Principal point of camera for pinhole projection model, in pixels.
        G_camera_image (:obj: `numpy.matrixlib.defmatrix.matrix`): Transform from image frame to camera frame.
        bilinear_lut (:obj: `numpy.ndarray`): Look-up table for undistortion of images, mapping pixels in an undistorted
            image to pixels in the distorted image

    """

    def __init__(self, models_dir, images_dir):
        """Loads a camera model from disk.

        Args:
            models_dir (str): directory containing camera model files.
            images_dir (str): directory containing images for which to read camera model.

        """
        self.camera = None
        self.camera_sensor = None
        self.focal_length = None
        self.principal_point = None
        self.G_camera_image = None
        self.bilinear_lut = None

        self.__load_intrinsics(models_dir, images_dir) # 加载相机内参
        self.__load_lut(models_dir, images_dir) # 加载畸变矫正表

    def project(self, xyz, image_size):
        """Projects a pointcloud into the camera using a pinhole camera model.

        Args:
            xyz (:obj: `numpy.ndarray`): 3xn array, where each column is (x, y, z) point relative to camera frame.
            image_size (tuple[int]): dimensions of image in pixel

        Returns:
            numpy.ndarray: 2xm array of points, where each column is the (u, v) pixel coordinates of a point in pixels.
            numpy.array: array of depth values for points in image.

        Note:
            Number of output points m will be less than or equal to number of input points n, as points that do not
            project into the image are discarded.

        """
        if xyz.shape[0] == 3: # 如果输入的是3维的点云
            xyz = np.stack((xyz, np.ones((1, xyz.shape[1])))) # 将点云转换为齐次坐标 4xn
        xyzw = np.linalg.solve(self.G_camera_image, xyz) # 公式为：G_camera_image * xyzw = xyz  G_camera_image图像到相机的变换矩阵，这里是将点云从图像坐标系转换到相机坐标系

        # Find which points lie in front of the camera
        in_front = [i for i in range(0, xyzw.shape[1]) if xyzw[2, i] >= 0] # 在相机坐标系下，如果z坐标小于0，说明这个点在相机后面，不在相机前面
        xyzw = xyzw[:, in_front] # 只保留在相机前面的点

        uv = np.vstack((self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0], # 公式为：u = fx * x / z + cx
                        self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1])) # 针孔模型的投影公式

        in_img = [i for i in range(0, uv.shape[1])
                  if 0.5 <= uv[0, i] <= image_size[1] and 0.5 <= uv[1, i] <= image_size[0]] # 判断投影点是否在图像内

        return uv[:, in_img], np.ravel(xyzw[2, in_img]) # 返回投影点的坐标和深度

    def undistort(self, image): 
        """Undistorts an image.

        Args:
            image (:obj: `numpy.ndarray`): A distorted image. Must be demosaiced - ie. must be a 3-channel RGB image.

        Returns:
            numpy.ndarray: Undistorted version of image.

        Raises:
            ValueError: if image size does not match camera model.
            ValueError: if image only has a single channel.

        """
        if image.shape[0] * image.shape[1] != self.bilinear_lut.shape[0]: # 判断图像的大小是否和相机模型的大小一致
            raise ValueError('Incorrect image size for camera model') 

        lut = self.bilinear_lut[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1])) # lut 2xHxW, 2表示x和y坐标，H和W分别表示图像的高和宽

        if len(image.shape) == 1:
            raise ValueError('Undistortion function only works with multi-channel images')

        # np.rollaxis函数的作用是将数组的轴向进行循环移动，比如原来的数组是3维的，shape为（2,3,4），axis=0，start=2，那么输出的数组的shape为（4,2,3）
        # 这里的作用是将图像的通道数放到最后一维，方便后面的reshape
        # map_coordinates函数的作用是根据给定的索引值，在输入数组中取值，这里的输入数组是lut，索引值是image，输出的数组的shape为（H,W,C）, order=1表示使用双线性插值
        undistorted = np.rollaxis(np.array([map_coordinates(image[:, :, channel], lut, order=1)
                                for channel in range(0, image.shape[2])]), 0, 3) 

        return undistorted.astype(image.dtype)


    def __get_model_name(self, images_dir):
        """Gets the name of the camera model from the images directory.
        
        Args:
            images_dir (str): directory containing images for which to read camera model.

        Returns:
            str: name of camera model.
        """
        self.camera = re.search('(stereo|mono_(left|right|rear))', images_dir).group(0) # 获取相机名称
        if self.camera == 'stereo':
            self.camera_sensor = re.search('(left|centre|right)', images_dir).group(0) # 获取相机传感器名称
            if self.camera_sensor == 'left': 
                return 'stereo_wide_left'
            elif self.camera_sensor == 'right':
                return 'stereo_wide_right'
            elif self.camera_sensor == 'centre':
                return 'stereo_narrow_left'
            else: # 未知相机传感器
                raise RuntimeError('Unknown camera model for given directory: ' + images_dir)
        else: # 单目相机
            return self.camera

    def __load_intrinsics(self, models_dir, images_dir):
        model_name = self.__get_model_name(images_dir)  # 获取相机模型名称
        intrinsics_path = os.path.join(models_dir, model_name + '.txt') # 获取相机内参文件路径

        with open(intrinsics_path) as intrinsics_file: # 读取相机内参文件 
            vals = [float(x) for x in next(intrinsics_file).split()] # 读取第一行
            self.focal_length = (vals[0], vals[1]) # 相机焦距 f_x, f_y
            self.principal_point = (vals[2], vals[3]) # 相机光心 c_x, c_y

            G_camera_image = [] # 相机到图像的变换矩阵
            for line in intrinsics_file:
                G_camera_image.append([float(x) for x in line.split()])
            self.G_camera_image = np.array(G_camera_image)

    def __load_lut(self, models_dir, images_dir):
        model_name = self.__get_model_name(images_dir) # 获取相机模型名称
        lut_path = os.path.join(models_dir, model_name + '_distortion_lut.bin') # 获取畸变矫正表路径

        lut = np.fromfile(lut_path, np.double) # 读取畸变矫正表  
        lut = lut.reshape([2, lut.size // 2]) # 重塑畸变矫正表 
        self.bilinear_lut = lut.transpose() # 转置畸变矫正表 nx2

