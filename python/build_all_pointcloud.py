from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from multiprocessing import cpu_count
import sys
import os
import open3d
import numpy as np
import time


if __name__ == "__main__":
    data_dir = '/home/cxd/mnt_data/common-datasets-c/Oxford_Robotcar_For_PointNetVLAD' # path to the dataset containing the runs dir
    pointcloud_save_dir = '/home/cxd/mnt_data/RobotCar-PointCloud' # path to the directory where the pointclouds will be saved
    runs_dirs = next(os.walk(data_dir))[1] # get all runs dir names
    lidar_type = 'lms_front'
    poses_type = 'ins'
    extrinsics_dir = '../extrinsics' # path to the extrinsics dir
    workers = cpu_count() # number of workers
    total_time = 0
    for run in runs_dirs:
        start_time = time.time()
        print('Building pointcloud for run: ', run)
        laser_path = os.path.join(data_dir, run, lidar_type) # full path to run dir
        poses_path = os.path.join(data_dir, run, 'gps', poses_type+'.csv') # full path to poses file
        save_path = os.path.join(pointcloud_save_dir, lidar_type) # full path to save pointcloud
        if not os.path.exists(save_path): # create save dir if it doesn't exist
            os.makedirs(save_path)
        timestamps_path = os.path.join(laser_path, os.pardir, lidar_type + '.timestamps') # full path to timestamps file
        with open(timestamps_path) as timestamps_file:
            start_timestamp = int(next(timestamps_file).split(' ')[0])
            end_timestamp = int(next(reversed(list(timestamps_file))).split(' ')[0])
        pointcloud, reflectance = build_pointcloud(laser_path, poses_path, extrinsics_dir, workers, start_timestamp, end_timestamp)
        # check if reflectance is empty
        if reflectance is not None:
            colours = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min()) # normalise reflectance
            colours = 1 / (1 + np.exp(-10 * (colours - colours.mean()))) # sigmoid normalisation
        else:
            colours = 'gray'
        # save pointcloud and reflectance to disk
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(-np.ascontiguousarray(pointcloud[[1, 0, 2]].transpose().astype(np.float64)))
        pcd.colors = open3d.utility.Vector3dVector(np.tile(colours[:, np.newaxis], (1, 3)).astype(np.float64))
        # Rotate pointcloud to align displayed coordinate frame colouring
        pcd.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))
        open3d.io.write_point_cloud(os.path.join(save_path, run + '.pcd'), pcd)
        end_time = time.time()
        print(f'Pointcloud {run} done in {end_time - start_time} seconds, saved to: {os.path.join(save_path, run + ".pcd")}')
        total_time += end_time - start_time

    print(f'Pointclouds built in {total_time} seconds')

