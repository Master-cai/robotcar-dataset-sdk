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

from ast import arg
from cProfile import run
import time
import os
import sys
import re
import numpy as np
from tqdm import tqdm

from transform import build_se3_transform
from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from velodyne import load_velodyne_raw, load_velodyne_binary, velodyne_raw_to_pointcloud
from multiprocessing import Process, Queue, cpu_count


def worker(lidar_dir, lidar, poses, timestamps, reflectance, G_posesource_laser, queue, p_bar_queue):
    """ Worker process to load and transform lidar data.
    Args:
        lidar_dir (str): path to directory containing lidar data
        lidar (str): lidar type
        poses (list[numpy.matrixlib.defmatrix.matrix]): list of SE3 poses in ins frame
        timestamps (list[float]): list of timestamps for poses
        reflectance: reflectance in pointcloud
        G_posesource_laser (numpy.matrixlib.defmatrix.matrix): SE3 transform from ins to lidar frame; G_ins_laser
        queue (multiprocessing.Queue): queue to put pointclouds into
        p_bar_queue (multiprocessing.Queue): queue to put progress bar updates into
    return: 
        None        
    """
    pointcloud = np.array([[0], [0], [0], [0]]) # 用于存储点云数据
    for i in range(0, len(poses)): # 遍历每一帧的位姿
        scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.bin') # 拼接点云文件路径
        if "velodyne" not in lidar: # 如果不是velodyne雷达
            if not os.path.isfile(scan_path): 
                continue 

            scan_file = open(scan_path) # 打开点云文件
            scan = np.fromfile(scan_file, np.double) # 读取点云数据
            scan_file.close() # 关闭文件
            # 为什么不直接转为(3, n),而是使用reshape转换为(3, n)再转置为(n, 3)？
            scan = scan.reshape((len(scan) // 3, 3)).transpose() # 转换为3xN的矩阵

            if lidar != 'ldmrs': # 如果不是ldmrs雷达，存在反射率
                # LMS scans are tuples of (x, y, reflectance)
                # np.ravel() 将多维数组降为一维
                reflectance = np.concatenate((reflectance, np.ravel(scan[2, :]))) # 将反射率数据拼接到reflectance中
                scan[2, :] = np.zeros((1, scan.shape[1])) # 将反射率置为0，即：2D雷达，z坐标为0
        else: # 如果是velodyne雷达
            if os.path.isfile(scan_path):
                ptcld = load_velodyne_binary(scan_path)
            else:
                scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.png')
                if not os.path.isfile(scan_path):
                    continue
                ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(scan_path)
                ptcld = velodyne_raw_to_pointcloud(ranges, intensities, angles)

            reflectance = np.concatenate((reflectance, ptcld[3]))
            scan = ptcld[:3]
            
        # 将点云数据转换到ins坐标系下, 公式为：G_ins_laser * G_laser_ins * scan
        scan = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))])) # np.vstack([scan, np.ones((1, scan.shape[1]))])是将scan转换为4xN的矩阵，因为G_laser_ins是4x4的矩阵
        pointcloud = np.hstack([pointcloud, scan])
        p_bar_queue.put(1)

    pointcloud = pointcloud[:, 1:] # remove the first column  (0, 0, 0, 0)
    if pointcloud.shape[1] == 0:
        raise IOError("Could not find scan files for given time range in directory " + lidar_dir)
    queue.put((pointcloud, reflectance))



def build_pointcloud(lidar_dir, poses_file, extrinsics_dir, workers, start_time, end_time, origin_time=-1):
    """Builds a pointcloud by combining multiple LIDAR scans with odometry information.

    Args:
        lidar_dir (str): Directory containing LIDAR scans.
        poses_file (str): Path to a file containing pose information. Can be VO or INS data.
        extrinsics_dir (str): Directory containing extrinsic calibrations.
        workers (int): Number of workers to use for processing.
        start_time (int): UNIX timestamp of the start of the window over which to build the pointcloud.
        end_time (int): UNIX timestamp of the end of the window over which to build the pointcloud.
        origin_time (int): UNIX timestamp of origin frame. Pointcloud coordinates are relative to this frame.

    Returns:
        numpy.ndarray: 3xn array of (x, y, z) coordinates of pointcloud
        numpy.array: array of n reflectance values or None if no reflectance values are recorded (LDMRS)

    Raises:
        ValueError: if specified window doesn't contain any laser scans.
        IOError: if scan files are not found.

    """
    if origin_time < 0:
        origin_time = start_time

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', lidar_dir).group(0)
    timestamps_path = os.path.join(lidar_dir, os.pardir, lidar + '.timestamps')

    timestamps = []
    with open(timestamps_path) as timestamps_file:
        for line in timestamps_file:
            timestamp = int(line.split(' ')[0])
            if start_time <= timestamp <= end_time:
                timestamps.append(timestamp)

    if len(timestamps) == 0:
        raise ValueError("No LIDAR data in the given time bracket.")

    with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    # body是车体坐标系, 以stereo camera为原点, x轴指向前, y轴指向左, z轴指向上
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')]) # G_posesource_laser 是 lidar 到 body 的变换；posesource代表 body

    poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1) # vo, ins, rtk;

    if poses_type in ['ins', 'rtk']: # ins 和 rtk 都是用的 ins 的插值
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file) # 读取第一行 这里更新了extrinsics 为 ins 的 extrinsics，所以后面的插值是基于 ins 的
            # G_posesource_laser是 lidar 到 imu 的变换矩阵; G_laser_ins 是 imu 到 ins 的变换矩阵; G_ins_posesource 是 ins 到 lidar 的变换矩阵
            # 这里命名有点问题，应该是 G_ins_laser
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]), # G_posesource_laser = G_laser_ins * G_ins_posesource
                                                 G_posesource_laser)
        # 对 ins 进行插值，得到每个 lidar scan 的 ins 位姿；poses是在 ins 坐标系下的位姿
        poses = interpolate_ins_poses(poses_file, timestamps, origin_time, use_rtk=(poses_type == 'rtk')) # 用 ins 的插值 生成 poses，origin_time 是 start_time
    else: # 使用 vo 位姿
        # 对 vo 进行插值，得到每个 lidar scan 的 vo 位姿
        poses = interpolate_vo_poses(poses_file, timestamps, origin_time)

    pointcloud = np.array([[0], [0], [0], [0]])
    if lidar == 'ldmrs': # ldmrs没有反射率
        reflectance = None
    else: # 其他的有反射率
        reflectance = np.empty((0))

    # 多进程处理
    process_list= []
    queue = Queue() # 用于存储每个进程的结果
    p_bar_queue = Queue()  # 用于存储每个进程的进度
    workers = min(cpu_count(), 32) # 最多使用32个进程
    print("use %d workers to build point cloud..." %workers) # 打印使用的进程数
    step = len(poses) // workers + 1 # 每个进程处理的帧数
    for i in range(0, len(poses), step): # 每个进程处理的帧数
        process_list.append(Process(target=worker, args=(lidar_dir, lidar, poses[i: min(i+step, len(poses))], timestamps[i: min(i+step, len(poses))], reflectance, G_posesource_laser, queue, p_bar_queue))) # 创建进程
        process_list[-1].start() # 启动进程
        
    # 主线程处理进度条
    with tqdm(total=len(poses)) as pbar:
        for i in range(len(poses)):
            p_bar_queue.get()
            pbar.update(1)

    # 主线程处理结果
    for pre in range(len(process_list)):
        p, r = queue.get()
        pointcloud = np.hstack([pointcloud, p])
        reflectance = np.concatenate((reflectance, r))

    # 等待所有进程结束
    for process in process_list:
        process.join()
    print("all done")
    return pointcloud, reflectance


if __name__ == "__main__":
    import argparse
    import open3d

    parser = argparse.ArgumentParser(description='Build and display a pointcloud')
    parser.add_argument('--poses_file', type=str, default=None, help='File containing relative or absolute poses')
    parser.add_argument('--extrinsics_dir', type=str, default=None,
                        help='Directory containing extrinsic calibrations')
    parser.add_argument('--laser_dir', type=str, default=None, help='Directory containing LIDAR data')
    parser.add_argument('--start_timestamp', type=str, default=None, help='timestamp of the first LiDAR scan to build point cloud')
    parser.add_argument('--end_timestamp', type=str, default=None, help='timestamp of the last LiDAR scan to build point cloud')
    parser.add_argument('--workers', type=int, default=min(cpu_count(), 32), help='number of workers in to build point cloud')
    parser.add_argument('--visualization', action='store_true', help='visualize the point cloud')

    args = parser.parse_args()

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', args.laser_dir).group(0)
    timestamps_path = os.path.join(args.laser_dir, os.pardir, lidar + '.timestamps')
    if args.start_timestamp:
        start_timestamp = int(args.start_timestamp)
    else:
        with open(timestamps_path) as timestamps_file:
            start_timestamp = int(next(timestamps_file).split(' ')[0])

    if args.end_timestamp:
        end_timestamp = int(args.end_timestamp)
    else:
        end_timestamp = start_timestamp + 2e6  # 2 seconds
    
    start_time = time.time()
    pointcloud, reflectance = build_pointcloud(args.laser_dir, args.poses_file,
                                               args.extrinsics_dir, args.workers, start_timestamp, end_timestamp)
    end_time = time.time()
    print("time to build pointcloud: %d s" %(end_time - start_time))

    if reflectance is not None:
        colours = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min())
        colours = 1 / (1 + np.exp(-10 * (colours - colours.mean())))
    else:
        colours = 'gray'

    # 
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(
        -np.ascontiguousarray(pointcloud[[1, 0, 2]].transpose().astype(np.float64))) # 交换 x y 轴，负号是为了和 rviz 一致
    pcd.colors = open3d.utility.Vector3dVector(np.tile(colours[:, np.newaxis], (1, 3)).astype(np.float64))
    # Rotate pointcloud to align displayed coordinate frame colouring
    pcd.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))
    open3d.io.write_point_cloud('test_full.pcd', pcd)

    # Pointcloud Visualisation using Open3D
    # vis = open3d.Visualizer()
    if args.visualization:
        vis = open3d.visualization.Visualizer()
        run_name = os.path.basename(os.path.dirname(args.laser_dir))
        vis.create_window(window_name=run_name)
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
        # render_option.point_color_option = open3d.PointColorOption.ZCoordinate
        render_option.point_color_option = open3d.visualization.PointColorOption.ZCoordinate
        # coordinate_frame = open3d.geometry.create_mesh_coordinate_frame()
        coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame()
        vis.add_geometry(coordinate_frame)
        vis.add_geometry(pcd)
        view_control = vis.get_view_control()
        params = view_control.convert_to_pinhole_camera_parameters()
        params.extrinsic = build_se3_transform([0, 3, 10, 0, -np.pi * 0.42, -np.pi / 2])
        view_control.convert_from_pinhole_camera_parameters(params)
        vis.run()
