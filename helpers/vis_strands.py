#!/usr/bin/env python3
import os
from argparse import ArgumentParser
import numpy as np
import torch
import pickle as pkl
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation, RotationSpline
import numpy
from pytorch3d.io import load_obj, save_ply
import cv2

# -------------------- Linear algebra helpers (unchanged) --------------------

def rf_rq(P):
    P = P.T
    q, r = numpy.linalg.qr(P[::-1, ::-1], 'complete')
    q = q.T
    q = q[::-1, ::-1]
    r = r.T
    r = r[::-1, ::-1]
    if (numpy.linalg.det(q) < 0):
        r[:, 0] *= -1
        q[0, :] *= -1
    return r, q

def KRT_from_P(P):
    N = 3
    H = P[:, 0:N]
    K, R = rf_rq(H)
    K /= K[-1, -1]
    sg = numpy.diag(numpy.sign(numpy.diag(K)))
    K = K @ sg
    R = sg @ R
    if (numpy.linalg.det(R) < 0):
        R = -R
    C = numpy.linalg.lstsq(-H, P[:, -1], rcond=None)[0]
    T = -R @ C
    return K, R, T

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

# -------------------- Hair OBJ parsing --------------------

def parse_obj_polylines(obj_path):
    """
    Preferred: OBJ with 'v' vertices and 'l' polyline faces referencing vertex indices (1-based).
    Returns: list of (Mi,3) float32 arrays (variable lengths).
    """
    verts = [None]  # align with 1-indexing
    strands = []
    with open(obj_path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                x, y, z = map(float, parts[1:4])
                verts.append((x, y, z))
            elif tag == "l" and len(parts) >= 3:
                idxs = []
                for tok in parts[1:]:
                    # handle possible "i/j/k" tokens by taking the first part
                    idxs.append(int(tok.split("/")[0]))
                strand = np.array([verts[i] for i in idxs], dtype=np.float32)
                if len(strand) > 0:
                    strands.append(strand)
    return strands

def parse_obj_vertex_blocks(obj_path):
    """
    Fallback: Some OBJs store each strand as a separate object/group or block of 'v' lines.
    We treat blank lines and 'o'/'g' headers as boundaries.
    """
    strands, cur = [], []
    with open(obj_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if cur:
                    strands.append(np.array(cur, dtype=np.float32))
                    cur = []
                continue
            parts = line.split()
            tag = parts[0]
            if tag in ("o", "g"):
                if cur:
                    strands.append(np.array(cur, dtype=np.float32))
                    cur = []
                continue
            if tag == "v" and len(parts) >= 4:
                x, y, z = map(float, parts[1:4])
                cur.append((x, y, z))
            else:
                # any other statement ends current block if non-empty
                if cur:
                    strands.append(np.array(cur, dtype=np.float32))
                    cur = []
    if cur:
        strands.append(np.array(cur, dtype=np.float32))
    return [s for s in strands if len(s) > 0]

def load_hair_strands_from_obj(obj_path, target_len, chunk_if_needed=True):
    """
    Load strands from OBJ (polylines preferred; else block/grouped vertices).
    Returns a (N, target_len, 3) array after padding/truncation.
    """
    strands = parse_obj_polylines(obj_path)
    if len(strands) == 0:
        strands = parse_obj_vertex_blocks(obj_path)

    if len(strands) == 0:
        if not chunk_if_needed:
            raise ValueError(f"No strands could be parsed from {obj_path}.")
        # Final fallback: chunk all vertices into fixed-length strands
        all_v = []
        with open(obj_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    _, xs, ys, zs = line.strip().split()[:4]
                    all_v.append((float(xs), float(ys), float(zs)))
        if len(all_v) == 0:
            raise ValueError(f"No vertices found in {obj_path}.")
        if len(all_v) <= target_len:
            strands = [np.array(all_v, dtype=np.float32)]
        else:
            strands = []
            for i in range(0, len(all_v), target_len):
                strands.append(np.array(all_v[i:i+target_len], dtype=np.float32))

    # pad/truncate to target_len
    fixed = []
    for s in strands:
        if len(s) >= target_len:
            fixed.append(s[:target_len])
        else:
            pad = np.repeat(s[-1][None, :], target_len - len(s), axis=0)
            fixed.append(np.concatenate([s, pad], axis=0))
    hair = np.stack(fixed, axis=0).astype(np.float32)  # (N, L, 3)
    return hair

# -------------------- Main pipeline (no bpy here) --------------------

def main(blender_path, input_path, exp_name_1, exp_name_3,
         strand_length, speed_up, max_frames,
         hair_obj, head_obj=None, strand_radius=None):
    out_dir = f'{input_path}/curves_reconstruction/{exp_name_3}/blender'
    os.makedirs(f'{out_dir}/results', exist_ok=True)

    # Gather frames
    img_dir = f'{input_path}/images_2'
    frames = [
        int(fname.split('.')[0])
        for fname in sorted(os.listdir(img_dir))
        if fname.split('.')[0].isdigit()
    ]
    if len(frames) < 2:
        raise RuntimeError(f"Need at least two numeric frames in {img_dir} to interpolate cameras.")

    # Load camera dict (same as before)
    cam_pkl = f'{input_path}/3d_gaussian_splatting/{exp_name_1}/cameras/30000_matrices.pkl'
    cameras = pkl.load(open(cam_pkl, 'rb'))

    # Build per-frame K,R,T and interpolate
    R_list, K_list, T_list = [], [], []
    for frame in frames:
        scale_x, scale_y = 1080, 1920
        P34 = cameras['%06d' % frame].transpose(0, 1)[:3, :4].numpy()
        intrinsics, pose = load_K_Rt_from_P(None, P34)
        pose_all_inv = np.linalg.inv(pose)
        intrinsics_modified = intrinsics.copy()
        intrinsics_modified[0, 0] /= 2
        intrinsics_modified[1, 1] /= 2
        intrinsics_modified[0, 2] /= 2
        intrinsics_modified[1, 2] /= 2
        intrinsics_modified[0, 2] += 0.5
        intrinsics_modified[1, 2] += 0.5
        scaling_matrix = np.array([[scale_x, 0, 0, 0],
                                   [0, scale_y, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
        projection_matrix = scaling_matrix @ intrinsics_modified @ pose_all_inv
        K_w2c, R_w2c, T_w2c = KRT_from_P(projection_matrix[:3])
        R_list.append(R_w2c); K_list.append(K_w2c); T_list.append(T_w2c)

    rotations = Rotation.from_matrix(np.stack(R_list))
    spline = RotationSpline(frames, rotations)
    R_interp = spline(list(range(frames[-1]))).as_matrix()

    cameras_interp = []
    prev_j = -1
    next_j = 0
    for i in range(frames[-1]):
        if i in frames:
            prev_j += 1
            next_j += 1
        prev_K, prev_T = K_list[prev_j], T_list[prev_j]
        next_K, next_T = K_list[next_j], T_list[next_j]
        denom = max(1, (frames[next_j] - frames[prev_j]))
        alpha = 1 - (i - frames[prev_j]) / denom
        K_cur = prev_K * alpha + next_K * (1 - alpha)
        T_cur = prev_T * alpha + next_T * (1 - alpha)
        cameras_interp.append(K_cur @ np.concatenate([R_interp[i], T_cur[:, None]], axis=1))

    cameras_interp = np.stack(cameras_interp)[frames[0]:frames[-1]:speed_up][:max_frames]
    np.save(f'{out_dir}/cameras.npy', cameras_interp)

    # Head mesh -> PLY (use provided head_obj if given, otherwise your FLAME mesh)
    if head_obj is None:
        head_obj = f'{input_path}/flame_fitting/{exp_name_1}/stage_3/mesh_final.obj'
    verts, faces, _ = load_obj(head_obj)
    save_ply(f'{out_dir}/raw_head.ply', verts=verts, faces=faces.verts_idx)

    # Repack head PLY to vertex-only + face element, mirroring your original flow
    head_ply = PlyData.read(f'{out_dir}/raw_head.ply')
    head_vertex = (
        np.stack([
            head_ply.elements[0].data['x'],
            head_ply.elements[0].data['y'],
            head_ply.elements[0].data['z']], axis=1).reshape(-1, 3, 1)
    )[..., 0]
    head_vertex = [tuple(vtx) for vtx in head_vertex.tolist()]
    head_vertex = np.array(head_vertex, dtype=np.dtype('float, float, float'))
    head_vertex.dtype.names = ['x', 'y', 'z']
    head_new_ply = PlyData([PlyElement.describe(head_vertex, 'vertex'), head_ply.elements[1]])
    head_new_ply.write(f'{out_dir}/head.ply')

    # Hair strands from OBJ -> hair.npy (N, L, 3), with your axis remap [x, -z, y]
    hair = load_hair_strands_from_obj(hair_obj, target_len=strand_length, chunk_if_needed=True)  # (N,L,3)
    hair = np.stack([hair[..., 0], -hair[..., 2], hair[..., 1]], axis=-1).astype(np.float32)
    np.save(f'{out_dir}/hair.npy', hair)

    # Build Blender command (no bpy here)
    cmd = (
        f'{blender_path} -b main.blend -P render_color.py -- --args '
        f'{out_dir}/cameras.npy '
        f'{out_dir}/head.ply '
        f'{out_dir}/hair.npy '
        f'{out_dir}/results_perm '
        f'128 {frames[0]} {speed_up}'
    )
    if strand_radius is not None:
        cmd += f' --strand_radius {strand_radius}'
    os.system(cmd)

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--blender_path', type=str, required=True,
                        help='Path to Blender executable (e.g., /path/to/blender)')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Dataset root with images_2/ and 3d_gaussian_splatting/')
    parser.add_argument('--exp_name_1', default='stage1_lor=0.1', type=str)
    parser.add_argument('--exp_name_3', default='stage3_lor=0.1', type=str)

    # Geometry I/O
    parser.add_argument('--hair_obj', type=str, required=True,
                        help='OBJ file containing hair strands (supports "l" polylines or grouped vertex blocks).')
    parser.add_argument('--head_obj', type=str, default=None,
                        help='Optional head OBJ (defaults to FLAME mesh_final.obj if omitted).')

    # Strand handling
    parser.add_argument('--strand_length', default=100, type=int,
                        help='Target strand length; strands padded/truncated to this length.')

    # Rendering cadence
    parser.add_argument('--speed_up', default=4, type=int)
    parser.add_argument('--max_frames', default=200, type=int)

    # Optional thickness override (None => keep prior Blender default)
    parser.add_argument('--strand_radius', type=float, default=None,
                        help='Optional override for hair thickness in Blender scene units.')

    args = parser.parse_args()

    main(args.blender_path, args.input_path, args.exp_name_1, args.exp_name_3,
         args.strand_length, args.speed_up, args.max_frames,
         args.hair_obj, args.head_obj, args.strand_radius)