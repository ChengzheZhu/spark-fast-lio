#!/usr/bin/env python3
"""Merge per-scan PCDs by a pose file into one world-frame PCD (for CloudCompare / QA).

Reads <data_path>/pcd/<i>.pcd (binary XYZI, as written by hba_bridge) + a pose file
(one line "tx ty tz qw qx qy qz" per scan; line i <-> pcd/i.pcd), transforms each scan to world
(p_w = R(q)*p + t), merges, optional voxel downsample, writes one binary XYZI PCD.

Usage: merge_map.py <data_path/> <pose_file> <out.pcd> [voxel_leaf=0]
  voxel_leaf 0 = full density; e.g. 0.005 = 5 mm.
"""
import os
import sys
import numpy as np

_PCD_HEADER = (
    "# .PCD v0.7 - Point Cloud Data\n"
    "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
    "WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
)
_MARK = b'DATA binary\n'


def read_pcd_xyzi(path):
    with open(path, 'rb') as f:
        data = f.read()
    idx = data.find(_MARK)
    if idx < 0:
        raise ValueError('not a binary PCD: ' + path)
    header = data[:idx].decode('ascii', 'ignore')
    npts, ncol = None, 4
    for line in header.splitlines():
        if line.startswith('POINTS'):
            npts = int(line.split()[1])
        elif line.startswith('FIELDS'):
            ncol = len(line.split()) - 1
    body = data[idx + len(_MARK):]
    return np.frombuffer(body, dtype='<f4', count=npts * ncol).reshape(npts, ncol)


def quat_to_R(qw, qx, qy, qz):
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def read_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            s = line.split()
            if len(s) < 7:
                continue
            tx, ty, tz, qw, qx, qy, qz = map(float, s[:7])
            poses.append((np.array([tx, ty, tz], dtype=np.float64), quat_to_R(qw, qx, qy, qz)))
    return poses


def write_pcd_xyzi(path, xyzi):
    with open(path, 'wb') as f:
        f.write(_PCD_HEADER.format(n=xyzi.shape[0]).encode('ascii'))
        f.write(np.ascontiguousarray(xyzi, dtype='<f4').tobytes())


def voxel_downsample(xyzi, leaf):
    if leaf <= 0:
        return xyzi
    keys = np.floor(xyzi[:, :3] / leaf).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyzi[np.sort(idx)]


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    data_path = sys.argv[1]
    if not data_path.endswith('/'):
        data_path += '/'
    pose_file, out = sys.argv[2], sys.argv[3]
    leaf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

    poses = read_poses(pose_file)
    chunks, total = [], 0
    for i, (t, R) in enumerate(poses):
        p = os.path.join(data_path, 'pcd', '%d.pcd' % i)
        if not os.path.exists(p):
            print('  %s missing -> stop at %d scans' % (p, i))
            break
        arr = read_pcd_xyzi(p)
        out_arr = np.empty((arr.shape[0], 4), dtype=np.float32)
        out_arr[:, :3] = arr[:, :3].astype(np.float64) @ R.T + t
        out_arr[:, 3] = arr[:, 3] if arr.shape[1] > 3 else 0.0
        chunks.append(out_arr)
        total += arr.shape[0]
        if i % 200 == 0:
            print('  merged scan %d (%d pts cum)' % (i, total))
    if not chunks:
        print('no scans merged'); sys.exit(1)
    merged = np.concatenate(chunks, axis=0)
    print('merged %d scans, %d points' % (len(chunks), merged.shape[0]))
    if leaf > 0:
        merged = voxel_downsample(merged, leaf)
        print('voxel %.3f m -> %d points' % (leaf, merged.shape[0]))
    write_pcd_xyzi(out, merged)
    print('wrote %s (%d pts)' % (out, merged.shape[0]))


if __name__ == '__main__':
    main()
