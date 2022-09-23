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

import argparse
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime as dt
from image import load_image
from camera_model import CameraModel

parser = argparse.ArgumentParser(description='Play back images from a given directory')

parser.add_argument('dir', type=str, help='Directory containing images.') 
parser.add_argument('--models_dir', type=str, default=None, help='(optional) Directory containing camera model. If supplied, images will be undistorted before display')
parser.add_argument('--scale', type=float, default=1.0, help='(optional) factor by which to scale images before display')

args = parser.parse_args()

camera = re.search('(stereo|mono_(left|right|rear))', args.dir).group(0) # 相机类型

timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, camera + '.timestamps')) # 时间戳文件，为什么嵌套一层os.path.join()？因为os.path.join()可以自动处理路径分隔符，不同系统的路径分隔符不同，比如windows是\，linux是/，os.path.join()会自动处理
if not os.path.isfile(timestamps_path): # 如果时间戳文件不存在
  timestamps_path = os.path.join(args.dir, os.pardir, os.pardir, camera + '.timestamps') # 再往上一层找
  if not os.path.isfile(timestamps_path): # 如果还是不存在
      raise IOError("Could not find timestamps file")

model = None # 相机模型
if args.models_dir: # 如果指定了相机模型文件夹
    model = CameraModel(args.models_dir, args.dir) # 加载相机模型

current_chunk = 0 # 当前chunk
timestamps_file = open(timestamps_path) # 打开时间戳文件
for line in timestamps_file: # 逐行读取
    tokens = line.split()
    datetime = dt.utcfromtimestamp(int(tokens[0])/1000000) # 时间戳， 为什么要除以1000000？因为时间戳是微秒级的，需要转换成秒级
    chunk = int(tokens[1]) 

    filename = os.path.join(args.dir, tokens[0] + '.png')
    if not os.path.isfile(filename):
        if chunk != current_chunk:
            print("Chunk " + str(chunk) + " not found")
            current_chunk = chunk
        continue

    current_chunk = chunk

    img = load_image(filename, model)
    plt.imshow(img) 
    plt.xlabel(datetime) # 在图片下方显示时间戳
    plt.xticks([]) # 不显示x轴刻度
    plt.yticks([]) # 不显示y轴刻度
    plt.pause(0.01) # 暂停0.01秒，这样可以看到每一帧图片的时间戳
