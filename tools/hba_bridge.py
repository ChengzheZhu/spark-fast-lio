#!/usr/bin/env python3
"""Bridge spark-fast-lio output -> HBA (hku-mars/HBA) input format.

While a recorded bag is replayed through spark-fast-lio (config `visualization_frame: lidar`,
`publish.scan_lidarframe_pub_en: true`), this node pairs each per-scan cloud with its pose and
writes the HBA layout:

    <out_dir>/
      pcd/0.pcd, 1.pcd, ...     # each = one undistorted scan in the LiDAR frame (PointXYZI)
      pose.json                 # one line per scan, "tx ty tz qw qx qy qz" (T_world_lidar)

Inputs (same header.stamp, paired via ApproximateTimeSynchronizer):
  /cloud_registered_lidar (sensor_msgs/PointCloud2) = spark's undistorted scan in the LiDAR frame
  /odometry               (nav_msgs/Odometry)       = T_world_lidar (because viz_frame=lidar)

Run (with ROS + ws_livox + spark installs sourced):
  python3 hba_bridge.py --ros-args -p out_dir:=$HOME/scans/hba/<seq>
then in other terminals launch spark and `ros2 bag play <bag>`. Ctrl-C when playback ends.
"""
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
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
    """xyzi: (N,4) float32 array -> binary_little_endian PCD."""
    with open(path, 'wb') as f:
        f.write(_PCD_HEADER.format(n=xyzi.shape[0]).encode('ascii'))
        f.write(np.ascontiguousarray(xyzi, dtype='<f4').tobytes())


class HbaBridge(Node):
    def __init__(self):
        super().__init__('hba_bridge')
        out_dir = self.declare_parameter(
            'out_dir', os.path.expanduser('~/scans/hba/run')).value
        self.every = max(1, int(self.declare_parameter('every', 1).value))  # keep 1 of every N scans
        self.pcd_dir = os.path.join(out_dir, 'pcd')
        os.makedirs(self.pcd_dir, exist_ok=True)
        self.pose_f = open(os.path.join(out_dir, 'pose.json'), 'w')
        self.idx = 0     # index of KEPT frames (pcd/<idx>.pcd + pose line)
        self.seen = 0    # count of all synced pairs

        qos = QoSProfile(depth=200,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        cloud_sub = Subscriber(self, PointCloud2, '/cloud_registered_lidar', qos_profile=qos)
        odom_sub = Subscriber(self, Odometry, '/odometry', qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer([cloud_sub, odom_sub], queue_size=200, slop=0.02)
        self.sync.registerCallback(self.on_pair)

        self.get_logger().info('hba_bridge -> %s (pcd/ + pose.json), every=%d. Waiting for scans...'
                               % (out_dir, self.every))

    def on_pair(self, cloud, odom):
        keep = (self.seen % self.every == 0)
        self.seen += 1
        if not keep:
            return
        arr = point_cloud2.read_points(
            cloud, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True)
        n = arr.shape[0]
        if n == 0:
            return
        xyzi = np.empty((n, 4), dtype=np.float32)
        xyzi[:, 0] = arr['x']
        xyzi[:, 1] = arr['y']
        xyzi[:, 2] = arr['z']
        xyzi[:, 3] = arr['intensity']
        write_pcd_xyzi(os.path.join(self.pcd_dir, '%d.pcd' % self.idx), xyzi)

        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        # HBA format: tx ty tz qw qx qy qz. Newline is a SEPARATOR, not a terminator
        # (no trailing newline) so HBA's while(!eof) read_pose doesn't dup the last line.
        prefix = '' if self.idx == 0 else '\n'
        self.pose_f.write('%s%.9f %.9f %.9f %.9f %.9f %.9f %.9f'
                          % (prefix, p.x, p.y, p.z, q.w, q.x, q.y, q.z))
        self.pose_f.flush()

        if self.idx % 20 == 0:
            self.get_logger().info('frame %d: %d pts' % (self.idx, n))
        self.idx += 1

    def destroy_node(self):
        try:
            self.pose_f.close()
        except Exception:  # noqa: BLE001
            pass
        self.get_logger().info('hba_bridge done: %d frames written' % self.idx)
        super().destroy_node()


def main():
    rclpy.init()
    node = HbaBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == '__main__':
    main()
