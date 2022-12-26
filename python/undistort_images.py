import argparse
import os
import re
from PIL import Image
from image import load_image
from camera_model import CameraModel
from multiprocessing import Process, Queue, cpu_count
from tqdm import tqdm


# data_dir = '/home/cxd/data/Oxford_Robotcar_For_PointNetVLAD/2014-06-26-09-31-18/stereo/centre'
# model = None # 相机模型
# model = CameraModel('../models', data_dir) # 加载相机模型
# file_name = os.path.join(data_dir, '1403775807403450.png')

# img = load_image(file_name, model)
# img = Image.fromarray(img)
# img.save('test.png')


def undistort_image(image_pathes, camera_model, save_path, queue):
    """
    undistort image using camera model

    params
    ------
    image_pathes: str
        path to images
    camera_model: CameraModel
        camera model
    save_path: str
        path to save undistorted image
    queue: Queue
        queue to synchronize processes
    return
    ------
    None
    """
    for image_path in image_pathes:
        run_name = image_path.split('/')[-4]
        image_name = os.path.basename(image_path)
        img = load_image(image_path, camera_model)
        img = Image.fromarray(img)
        img.save(os.path.join(save_path, run_name,'stereo', 'centre', image_name))
        queue.put(1)

def load_all_image_paths(BASE_DIR, SAVE_PATH):
    """
    load all image paths

    params
    ------
    BASE_DIR: str
        base directory of dataset

    return
    ------
    image_paths: list
        list of image paths
    """
    image_paths = []
    _, dirs, _ = next(os.walk(BASE_DIR))
    for d in dirs:
        image_dir = os.path.join(BASE_DIR, d, 'stereo', 'centre')
        save_path = os.path.join(SAVE_PATH, d, 'stereo', 'centre')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        _, _, files = next(os.walk(image_dir))
        for f in files:
            if f.endswith('.png'):
                image_paths.append(os.path.join(image_dir, f))
    return image_paths

if __name__ == '__main__':
    BASE_DIR = '/home/cxd/data/Oxford_Robotcar_For_PointNetVLAD'
    SAVE_PATH = '/home/cxd/data/Oxford_Robotcar_For_PointNetVLAD_Undistorted'
    all_image_paths = load_all_image_paths(BASE_DIR, SAVE_PATH)
    # all_image_paths = all_image_paths[0:100]
    print(all_image_paths[0])
    model = CameraModel('../models', 'centre/stereo') # 加载相机模型
    print('total image number: {}'.format(len(all_image_paths)))
    # print(image_paths[0])

    # 多进程处理
    process_list= []
    queue = Queue() # 用于存储每个进程的进度
    workers = min(cpu_count(), 32) # 最多使用32个进程
    print("use %d workers to undistort images..." %workers) # 打印使用的进程数
    step = len(all_image_paths) // workers + 1 # 每个进程处理的帧数
    for i in range(0, len(all_image_paths), step): # 每个进程处理的帧数
        process_list.append(Process(target=undistort_image, args=(all_image_paths[i: min(i+step, len(all_image_paths))], model, SAVE_PATH, queue))) # 创建进程
        process_list[-1].start() # 启动进程
        
    # 主线程处理进度条
    with tqdm(total=len(all_image_paths)) as pbar:
        for i in range(len(all_image_paths)):
            queue.get()
            pbar.update(1)
    # 等待所有进程结束
    for process in process_list:
        process.join()
    print("all done")

