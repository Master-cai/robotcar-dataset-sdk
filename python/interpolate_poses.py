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

import bisect
import csv
import numpy as np
import numpy.matlib as ml
from transform import *


def interpolate_vo_poses(vo_path, pose_timestamps, origin_timestamp):
    """Interpolate poses from visual odometry.

    Args:
        vo_path (str): path to file containing relative poses from visual odometry.
        pose_timestamps (list[int]): UNIX timestamps at which interpolated poses are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    """
    with open(vo_path) as vo_file:
        vo_reader = csv.reader(vo_file)
        headers = next(vo_file)

        vo_timestamps = [0]
        abs_poses = [ml.identity(4)] # 初始位姿为单位矩阵

        lower_timestamp = min(min(pose_timestamps), origin_timestamp) # 可以减1以防止第一个时间戳不在vo_timestamps中
        upper_timestamp = max(max(pose_timestamps), origin_timestamp) # 可以加1以防止最后一个时间戳不在vo_timestamps中

        for row in vo_reader: 
            timestamp = int(row[0])
            if timestamp < lower_timestamp:
                vo_timestamps[0] = timestamp
                continue

            vo_timestamps.append(timestamp)

            xyzrpy = [float(v) for v in row[2:8]]
            rel_pose = build_se3_transform(xyzrpy)
            abs_pose = abs_poses[-1] * rel_pose
            abs_poses.append(abs_pose)

            if timestamp >= upper_timestamp:
                break

    return interpolate_poses(vo_timestamps, abs_poses, pose_timestamps, origin_timestamp)


def interpolate_ins_poses(ins_path, pose_timestamps, origin_timestamp, use_rtk=False):
    """Interpolate poses from INS.

    Args:
        ins_path (str): path to file containing poses from INS.
        pose_timestamps (list[int]): UNIX timestamps at which interpolated poses are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    """
    with open(ins_path) as ins_file: # 读取ins文件
        ins_reader = csv.reader(ins_file)
        headers = next(ins_file)

        ins_timestamps = [0] # ins时间戳
        abs_poses = [ml.identity(4)] # 初始位姿为单位矩阵

        upper_timestamp = max(max(pose_timestamps), origin_timestamp) # 可以加1以防止最后一个时间戳不在ins_timestamps中
        
        # 
        for row in ins_reader: # 读取ins数据的每一行
            timestamp = int(row[0]) # ins时间戳
            ins_timestamps.append(timestamp) # 添加到ins时间戳列表中

            utm = row[5:8] if not use_rtk else row[4:7] # row[5:8]为ins的utm坐标，row[4:7]为rtk的utm坐标
            rpy = row[-3:] if not use_rtk else row[11:14] # row[-3:]为ins的rpy角，row[11:14]为rtk的rpy角
            xyzrpy = [float(v) for v in utm] + [float(v) for v in rpy] # xyzrpy为ins的xyzrpy角
            abs_pose = build_se3_transform(xyzrpy) # 由xyzrpy角构建SE3矩阵 是相对于传感器坐标系的位姿
            abs_poses.append(abs_pose) # 添加到abs_poses列表中

            if timestamp >= upper_timestamp: # 如果时间戳大于upper_timestamp，则跳出循环
                break

    ins_timestamps = ins_timestamps[1:] # 去掉第一个时间戳 因为第一个时间戳为0
    abs_poses = abs_poses[1:] # 去掉第一个位姿 因为第一个位姿为单位矩阵
    # pose_timestamps为需要插值的时间戳列表，ins_timestamps为ins时间戳列表，abs_poses为ins位姿列表
    # ins_timestamps和abs_poses是一一对应的
    return interpolate_poses(ins_timestamps, abs_poses, pose_timestamps, origin_timestamp)


def interpolate_poses(pose_timestamps, abs_poses, requested_timestamps, origin_timestamp):
    """Interpolate between absolute poses.

    Args:
        pose_timestamps (list[int]): Timestamps of supplied poses. Must be in ascending order.
        abs_poses (list[numpy.matrixlib.defmatrix.matrix]): SE3 matrices representing poses at the timestamps specified.
        requested_timestamps (list[int]): Timestamps for which interpolated timestamps are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    Raises:
        ValueError: if pose_timestamps and abs_poses are not the same length
        ValueError: if pose_timestamps is not in ascending order

    """
    requested_timestamps.insert(0, origin_timestamp) # 在requested_timestamps列表的第一个位置插入origin_timestamp,因为origin_timestamp也需要插值
    requested_timestamps = np.array(requested_timestamps) # 将列表转换为数组
    pose_timestamps = np.array(pose_timestamps) # 将列表转换为数组

    if len(pose_timestamps) != len(abs_poses): # 如果时间戳和位姿的数量不一致，则抛出异常
        raise ValueError('Must supply same number of timestamps as poses')

    abs_quaternions = np.zeros((4, len(abs_poses))) # 初始化四元数矩阵 4行len(abs_poses)列
    abs_positions = np.zeros((3, len(abs_poses))) # 初始化位置矩阵 3行len(abs_poses)列
    for i, pose in enumerate(abs_poses): # 遍历abs_poses列表
        if i > 0 and pose_timestamps[i-1] >= pose_timestamps[i]: # 如果前一个时间戳大于等于后一个时间戳，则抛出异常
            raise ValueError('Pose timestamps must be in ascending order')

        abs_quaternions[:, i] = so3_to_quaternion(pose[0:3, 0:3]) # 将pose的旋转矩阵转换为四元数
        abs_positions[:, i] = np.ravel(pose[0:3, 3]) # 将pose的平移矩阵转换为位置矩阵
    # bisect 模块实现了二分查找算法; bisect.bisect(a, x, lo=0, hi=len(a)) 在有序列表a中查找x的插入位置，lo和hi为查找范围
    upper_indices = [bisect.bisect(pose_timestamps, pt) for pt in requested_timestamps] # 返回requested_timestamps中每个时间戳在pose_timestamps中的插入位置
    lower_indices = [u - 1 for u in upper_indices] # 返回requested_timestamps中每个时间戳在pose_timestamps中的前一个位置

    # 以下为插值计算
    # 若upper_indices的最大值大于pose_timestamps的长度，说明requested_timestamps中有时间戳大于pose_timestamps中的最大时间戳
    # 则将upper_indices中大于len(pose_timestamps) - 1 的值设置为len(pose_timestamps) - 1
    if max(upper_indices) >= len(pose_timestamps): 
        upper_indices = [min(i, len(pose_timestamps) - 1) for i in upper_indices] # 将upper_indices中的每个元素与pose_timestamps的长度-1比较，取较小的值
    # fraction表示requested_timestamps中每个时间戳在pose_timestamps中的插入位置与前一个位置的比值
    # 插值公式：x = x1 + (x2 - x1) * fraction
    fractions = (requested_timestamps - pose_timestamps[lower_indices]) // \
                (pose_timestamps[upper_indices] - pose_timestamps[lower_indices])

    quaternions_lower = abs_quaternions[:, lower_indices] # 获取lower_indices对应的四元数
    quaternions_upper = abs_quaternions[:, upper_indices] # 获取upper_indices对应的四元数
    # 四元数插值公式及其推导过程见：
    # https://zhuanlan.zhihu.com/p/87418561

    # 四元数的点积表示两个四元数的夹角余弦值
    d_array = (quaternions_lower * quaternions_upper).sum(0) # 计算四元数的点积，代表四元数的夹角 cos(theta)
    # np.nonzero(d_array>=1) 返回d_array中大于等于1的元素的索引
    # 对于小于90度的四元素，可以使用四元数插值公式：q = q1 + (q2 - q1) * fraction
    linear_interp_indices = np.nonzero(d_array >= 1) # d_array >= 1 代表四元数的夹角小于等于90度，即四元数的点积大于等于1  
    
    sin_interp_indices = np.nonzero(d_array < 1) # 获取夹角大于90度的四元数的索引 theta > 90 

    scale0_array = np.zeros(d_array.shape) # 初始化插值系数数组
    scale1_array = np.zeros(d_array.shape) # 初始化插值系数数组

    scale0_array[linear_interp_indices] = 1 - fractions[linear_interp_indices] # 四元数的夹角小于等于90度，使用四元数插值公式
    scale1_array[linear_interp_indices] = fractions[linear_interp_indices] # 四元数的夹角小于等于90度，使用四元数插值公式

    theta_array = np.arccos(np.abs(d_array[sin_interp_indices])) # 获取四元数的夹角 theta

    # 四元数的夹角大于90度，使用球面线性插值公式
    scale0_array[sin_interp_indices] = np.sin((1 - fractions[sin_interp_indices]) * theta_array) / np.sin(theta_array)
    scale1_array[sin_interp_indices] = np.sin(fractions[sin_interp_indices] * theta_array) / np.sin(theta_array)

    negative_d_indices = np.nonzero(d_array < 0) # d_array < 0, 代表四元数的夹角大于180度，即四元数的点积小于0  
    scale1_array[negative_d_indices] = -scale1_array[negative_d_indices] # 四元数的夹角大于180度，需要将scale1_array中的元素取反
    # np.tile(scale0_array, (4, 1)) 将scale0_array沿着第0维复制4次，沿着第1维复制1次
    # 公式为：q = q1 + (q2 - q1) * fraction，q1 = quaternions_lower * scale0_array，q2 = quaternions_upper * scale1_array
    quaternions_interp = np.tile(scale0_array, (4, 1)) * quaternions_lower + np.tile(scale1_array, (4, 1)) * quaternions_upper 

    positions_lower = abs_positions[:, lower_indices] # 获取lower_indices对应的位置
    positions_upper = abs_positions[:, upper_indices] # 获取upper_indices对应的位置
    
    # 对于位置，使用线性插值公式：x = x1 + (x2 - x1) * fraction
    # 公式为：p = p1 + (p2 - p1) * fraction，p1 = positions_lower，p2 = positions_upper
    positions_interp = np.multiply(np.tile((1 - fractions), (3, 1)), positions_lower) \
                       + np.multiply(np.tile(fractions, (3, 1)), positions_upper)

    # 将四元数归一化
    poses_mat = ml.zeros((4, 4 * len(requested_timestamps)))

    # 这里是将四元数转换为旋转矩阵的过程，具体过程见：
    # https://zhuanlan.zhihu.com/p/45404840
    # 公式为：R1 = [1 - 2 * (q2^2 + q3^2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)]
    # poses_mat[0, 0::4]： 第4n列的元素全部赋值为1 - 2 * (q2^2 + q3^2)
    poses_mat[0, 0::4] = 1 - 2 * np.square(quaternions_interp[2, :]) - \
                         2 * np.square(quaternions_interp[3, :])
    # poses_mat[0, 1::4]： 第4n+1列的元素全部赋值为2 * (q1 * q2 - q0 * q3)
    poses_mat[0, 1::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) - \
                         2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    # poses_mat[0, 2::4]： 第4n+2列的元素全部赋值为2 * (q1 * q3 + q0 * q2)
    poses_mat[0, 2::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])
    # 公式为：R2 = [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1^2 + q3^2), 2 * (q2 * q3 - q0 * q1)]
    poses_mat[1, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) \
                         + 2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[1, 1::4] = 1 - 2 * np.square(quaternions_interp[1, :]) \
                         - 2 * np.square(quaternions_interp[3, :])
    poses_mat[1, 2::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])
    # 公式为：R3 = [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1^2 + q2^2)]
    poses_mat[2, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])
    poses_mat[2, 1::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])
    poses_mat[2, 2::4] = 1 - 2 * np.square(quaternions_interp[1, :]) - \
                         2 * np.square(quaternions_interp[2, :])

    poses_mat[0:3, 3::4] = positions_interp
    poses_mat[3, 3::4] = 1

    poses_mat = np.linalg.solve(poses_mat[0:4, 0:4], poses_mat) # poses_mat

    poses_out = [0] * (len(requested_timestamps) - 1) # 除去第一个pose，其他pose的数量
    for i in range(1, len(requested_timestamps)): # 从第二个pose开始
        poses_out[i - 1] = poses_mat[0:4, i * 4:(i + 1) * 4] # 将旋转矩阵转换为pose, pose是一个4x4的矩阵

    return poses_out # 返回除去第一个pose的其他pose
