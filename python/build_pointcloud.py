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
    pointcloud = np.array([[0], [0], [0], [0]])
    for i in range(0, len(poses)):
    # for i in tqdm.trange(len(poses)):
        scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.bin')
        if "velodyne" not in lidar:
            if not os.path.isfile(scan_path):
                continue

            scan_file = open(scan_path)
            scan = np.fromfile(scan_file, np.double)
            scan_file.close()

            scan = scan.reshape((len(scan) // 3, 3)).transpose() # 3 X len

            if lidar != 'ldmrs':
                # LMS scans are tuples of (x, y, reflectance)
                reflectance = np.concatenate((reflectance, np.ravel(scan[2, :]))) # 
                scan[2, :] = np.zeros((1, scan.shape[1]))
        else:
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

        scan = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))]))
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
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)

    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                 G_posesource_laser)

        poses = interpolate_ins_poses(poses_file, timestamps, origin_time, use_rtk=(poses_type == 'rtk'))
    else:
        # sensor is VO, which is located at the main vehicle frame
        poses = interpolate_vo_poses(poses_file, timestamps, origin_time)

    pointcloud = np.array([[0], [0], [0], [0]])
    if lidar == 'ldmrs':
        reflectance = None
    else:
        reflectance = np.empty((0))

    process_list= []
    queue = Queue()
    p_bar_queue = Queue()
    workers = min(cpu_count(), 32)
    print("use %d workers to build point cloud..." %workers)
    step = len(poses) // workers + 1
    for i in range(0, len(poses), step):
        process_list.append(Process(target=worker, args=(lidar_dir, lidar, poses[i: min(i+step, len(poses))], timestamps[i: min(i+step, len(poses))], reflectance, G_posesource_laser, queue, p_bar_queue)))
        process_list[-1].start()
        
    with tqdm(total=len(poses)) as pbar:
        for i in range(len(poses)):
            p_bar_queue.get()
            pbar.update(1)


    for pre in range(len(process_list)):
        p, r = queue.get()
        pointcloud = np.hstack([pointcloud, p])
        reflectance = np.concatenate((reflectance, r))

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
        end_timestamp = start_timestamp + 2e6
    
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
        -np.ascontiguousarray(pointcloud[[1, 0, 2]].transpose().astype(np.float64)))
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
