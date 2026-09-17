#!/usr/bin/env python3
"""Bridge spark-fast-lio output -> HBA (hku-mars/HBA) input format.

Interactive: run spark-fast-lio (+RViz) and `ros2 bag play` at normal (real-time) pace so spark
fully processes each scan; this node BUFFERS every processed scan + pose in memory. When the bag
finishes, press ENTER here to EXPORT the HBA layout:

    <out_dir>/
      pcd/0.pcd, 1.pcd, ...    # spark's IMU-deskewed per-scan cloud in the LiDAR frame (PointXYZI)
      pose.json                # one line per scan, "tx ty tz qw qx qy qz" (T_world_lidar)
      pose_initial.json        # backup (HBA's write_pose overwrites pose.json)

Inputs (paired by header.stamp, ApproximateTimeSynchronizer):
  /cloud_registered_lidar (sensor_msgs/PointCloud2) = spark cloud_undistort_ (deskewed, LiDAR frame)
  /odometry               (nav_msgs/Odometry)       = T_world_lidar (viz_frame=lidar)

Run (ROS + ws_livox + spark installs sourced):
  python3 hba_bridge.py --ros-args -p out_dir:=$HOME/scans/hba/<seq> [-p every:=1]
"""
import os
import shutil
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry

_PCD_HEADER = (
    "# .PCD v0.7 - Point Cloud Data\n"
    "VERSION 0.7\n"
    "FIELDS x y z intensity\n"
    "SIZE 4 4 4 4\n"
    "TYPE F F F F\n"
    "COUNT 1 1 1 1\n"
    "WIDTH {n}\n"
    "HEIGHT 1\n"
    "VIEWPOINT 0 0 0 1 0 0 0\n"
    "POINTS {n}\n"
    "DATA binary\n"
)


def write_pcd_xyzi(path, xyzi):
    with open(path, 'wb') as f:
        f.write(_PCD_HEADER.format(n=xyzi.shape[0]).encode('ascii'))
        f.write(np.ascontiguousarray(xyzi, dtype='<f4').tobytes())


class HbaBridge(Node):
    def __init__(self):
        super().__init__('hba_bridge')
        self.out_dir = self.declare_parameter(
            'out_dir', os.path.expanduser('~/scans/hba/run')).value
        self.every = max(1, int(self.declare_parameter('every', 1).value))
        self._lock = threading.Lock()
        self._frames = []   # (xyzi ndarray, (tx,ty,tz,qw,qx,qy,qz)) per kept scan
        self._seen = 0

        qos = QoSProfile(depth=400,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        cloud_sub = Subscriber(self, PointCloud2, '/cloud_registered_lidar', qos_profile=qos)
        odom_sub = Subscriber(self, Odometry, '/odometry', qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer([cloud_sub, odom_sub], queue_size=400, slop=0.02)
        self.sync.registerCallback(self.on_pair)

        self.get_logger().info(
            'hba_bridge BUFFERING /cloud_registered_lidar + /odometry (every=%d). '
            'Play the bag, then press ENTER to export -> %s' % (self.every, self.out_dir))

    def on_pair(self, cloud, odom):
        keep = (self._seen % self.every == 0)
        self._seen += 1
        if not keep:
            return
        arr = point_cloud2.read_points(
            cloud, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True)
        n = arr.shape[0]
        if n == 0:
            return
        xyzi = np.empty((n, 4), dtype=np.float32)
        xyzi[:, 0] = arr['x']; xyzi[:, 1] = arr['y']; xyzi[:, 2] = arr['z']; xyzi[:, 3] = arr['intensity']
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        with self._lock:
            self._frames.append((xyzi, (p.x, p.y, p.z, q.w, q.x, q.y, q.z)))
            k = len(self._frames)
        if k % 50 == 0:
            self.get_logger().info('buffered %d frames' % k)

    def export(self):
        with self._lock:
            frames = list(self._frames)
        if not frames:
            self.get_logger().warn('nothing buffered — nothing exported')
            return
        pcd_dir = os.path.join(self.out_dir, 'pcd')
        os.makedirs(pcd_dir, exist_ok=True)
        lines = []
        for i, (xyzi, pose) in enumerate(frames):
            write_pcd_xyzi(os.path.join(pcd_dir, '%d.pcd' % i), xyzi)
            lines.append('%.9f %.9f %.9f %.9f %.9f %.9f %.9f' % pose)  # HBA: tx ty tz qw qx qy qz
        pose_path = os.path.join(self.out_dir, 'pose.json')
        with open(pose_path, 'w') as f:
            f.write('\n'.join(lines))   # newline as separator, no trailing newline
        shutil.copyfile(pose_path, os.path.join(self.out_dir, 'pose_initial.json'))
        self.get_logger().info(
            'EXPORTED %d frames -> %s  (pcd/ + pose.json + pose_initial.json backup)'
            % (len(frames), self.out_dir))


def main():
    rclpy.init()
    node = HbaBridge()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        input('\n[hba_bridge] play the bag; when it finishes press ENTER to export...\n')
    except (EOFError, KeyboardInterrupt):
        pass
    node.export()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
