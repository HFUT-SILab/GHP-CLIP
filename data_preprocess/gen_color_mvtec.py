import argparse
import copy
import os

import cv2
import numpy as np
import open3d as o3d
import tifffile as tiff
import torch
from PIL import Image
from torchvision import transforms

# must match the settings used to generate 2d_rendering / 2d_3d_cor
WINDOW = 1024
IMG_SIZE = 336

# view conventions, selected by --views: b = 9 pure x rotations; a = 5 x rotations + 4 y rotations
VIEWS = {
    "b": ([-144, -108, -72.0, -36.0, 0, 36.0, 72.0, 108.0, 144], []),
    "a": ([-45.0, -15.0, 0, 15.0, 45.0], [-30, -15, 15, 30]),
}
VIEW_LIST_X, VIEW_LIST_Y = VIEWS["b"]

# render point size per class
POINT_SIZE = {
    "potato": 8, "tire": 8, "cable_gland": 12,
}


def read_tiff_organized_pc(path):
    return tiff.imread(path)


def resize_organized_pc(organized_pc, target_height=IMG_SIZE, target_width=IMG_SIZE,
                        tensor_out=False, mode="nearest"):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).float()
    torch_resized = torch.nn.functional.interpolate(
        torch_organized_pc, size=(target_height, target_width), mode=mode)
    if tensor_out:
        return torch_resized.squeeze(dim=0)
    return torch_resized.squeeze(dim=0).permute(1, 2, 0).numpy()


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1],
                                organized_pc.shape[2])


def matrix_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])


def project_points(vis, o3d_pc_r):
    """Recompute pixel coordinates to compare with the existing 2d_3d_cor."""
    control = vis.get_view_control()
    param = control.convert_to_pinhole_camera_parameters()
    intrinsic = param.intrinsic.intrinsic_matrix
    extrinsic = param.extrinsic
    r = cv2.Rodrigues(extrinsic[:3, :3])[0]
    t = extrinsic[:3, 3:]
    points_2d, _ = cv2.projectPoints(np.asarray(o3d_pc_r.points), r, t, intrinsic, None)
    return (points_2d[:, 0, :] * IMG_SIZE / WINDOW).astype(int).clip(0, IMG_SIZE - 1)


def render_color(o3d_pc_r, view_idx, color_save_path, point_size, light_on, ref_cor_dir):
    """Render one color view. Returns (points matching cor, total points), or None without a reference cor."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=WINDOW, height=WINDOW, visible=False)
    vis.add_geometry(o3d_pc_r)

    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([1, 1, 1])
    # lighting off by default to get pure albedo, complementing the gray shaded renders
    render_option.light_on = light_on
    render_option.point_size = point_size
    render_option.mesh_show_back_face = True

    check = None
    if ref_cor_dir is not None:
        ref_path = os.path.join(ref_cor_dir, f"view_{view_idx}_cor.npy")
        if os.path.exists(ref_path):
            ref = np.load(ref_path)
            mine = project_points(vis, o3d_pc_r)
            if len(ref) == len(mine):
                same = int(np.all(ref[:, :2] == mine, axis=1).sum())
                check = (same, len(mine))
            else:
                check = (-1, len(mine))  # point count mismatch
                print(f"      !! view {view_idx}: cor 行数={len(ref)} 本地点数={len(mine)}")

    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    post_transform = transforms.Resize(
        (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC)
    image = post_transform(image)
    image.save(f"{color_save_path}/view_{view_idx}.png", dpi=(2000, 2000))
    vis.destroy_window()
    return check


def get_mv_color_images(tiff_path, rgb_path, point_size, color_save_path,
                        ref_cor_dir, light_on):
    print(f"--- Processing: {tiff_path}")
    os.makedirs(color_save_path, exist_ok=True)

    # point processing must match the one used to generate 2d_3d_cor
    organized_pc = read_tiff_organized_pc(tiff_path)
    organized_pc = resize_organized_pc(organized_pc, IMG_SIZE, IMG_SIZE)
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc)
    nonzero_indices = np.nonzero(unorganized_pc[:, 2])[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]

    if len(unorganized_pc_no_zeros) == 0:
        print(f"    Warning: no valid points, skipping.")
        return []

    rgb = Image.open(rgb_path).convert("RGB")
    rgb = transforms.Resize((IMG_SIZE, IMG_SIZE),
                            interpolation=transforms.InterpolationMode.BICUBIC)(rgb)
    unorganized_rgb = (np.array(rgb) / 255.0).reshape(-1, 3)
    selected_colors = unorganized_rgb[nonzero_indices, :]

    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(unorganized_pc_no_zeros)
    o3d_pc.colors = o3d.utility.Vector3dVector(selected_colors)
    o3d_pc.transform(matrix_x(np.pi))
    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))

    checks = []
    view_idx = 0
    for v_x in VIEW_LIST_X:
        o3d_pc_o = copy.deepcopy(o3d_pc)
        R = o3d_pc.get_rotation_matrix_from_axis_angle(
            [np.radians(v_x), np.radians(0), np.radians(0)])
        o3d_pc_r = o3d_pc_o.rotate(R, center=o3d_pc.get_center())
        c = render_color(o3d_pc_r, view_idx, color_save_path, point_size, light_on, ref_cor_dir)
        checks.append(c)
        view_idx += 1
    for v_y in VIEW_LIST_Y:
        o3d_pc_o = copy.deepcopy(o3d_pc)
        R = o3d_pc.get_rotation_matrix_from_axis_angle(
            [np.radians(0), np.radians(v_y), np.radians(0)])
        o3d_pc_r = o3d_pc_o.rotate(R, center=o3d_pc.get_center())
        c = render_color(o3d_pc_r, view_idx, color_save_path, point_size, light_on, ref_cor_dir)
        checks.append(c)
        view_idx += 1

    valid = [c for c in checks if c is not None and c[0] >= 0]
    if valid:
        ratio = sum(s for s, _ in valid) / sum(n for _, n in valid)
        tag = "OK" if ratio > 0.99 else "MISMATCH"
        print(f"    [cor check] {ratio*100:.2f}% 像素与已有 2d_3d_cor 一致  => {tag}")
    print(f"--- Finished. {view_idx} color views -> {color_save_path}")
    return checks


def process_directory(directory_path, point_size, out_dirname, light_on, limit):
    n = 0
    for root, dirs, files in os.walk(directory_path):
        dirs.sort()
        files.sort()
        for file in files:
            if not file.endswith(".tiff"):
                continue
            if limit and n >= limit:
                return n
            tiff_path = os.path.join(root, file)
            file_id = os.path.splitext(file)[0]
            rgb_path = tiff_path.replace("xyz", "rgb").replace(".tiff", ".png")
            if not os.path.exists(rgb_path):
                print(f"    !! 缺少 rgb: {rgb_path}，跳过")
                continue

            color_save_path = os.path.join(root.replace("xyz", out_dirname), file_id)
            ref_cor_dir = os.path.join(root.replace("xyz", "2d_3d_cor"), file_id)
            if not os.path.isdir(ref_cor_dir):
                ref_cor_dir = None

            os.makedirs(color_save_path, exist_ok=True)
            get_mv_color_images(tiff_path, rgb_path, point_size, color_save_path,
                                ref_cor_dir, light_on)
            n += 1
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate multi-view true-color images for MVTec-3D", add_help=True)
    parser.add_argument("--data_path", type=str,
                        default="/root/autodl-tmp/mvtec_3d_anomaly_detection",
                        help="MVTec-3D 数据根目录（含 <cls>/test/*/xyz/*.tiff）")
    parser.add_argument("--out_dirname", type=str, default="2d_color",
                        help="输出目录名，与 2d_rendering 同级")
    parser.add_argument("--only_class", type=str, default=None,
                        help="只处理某一个类，用于先做校验")
    parser.add_argument("--limit", type=int, default=0,
                        help="每个类最多处理几个样本，0 表示不限")
    parser.add_argument("--light_on", action="store_true",
                        help="打开光照。默认关闭以得到纯 albedo，与灰度 shading 互补")
    parser.add_argument("--views", type=str, default="a", choices=["b", "a"],
                        help="视角约定。b=9 个纯 x 旋转；a=5x+4y")
    args = parser.parse_args()

    VIEW_LIST_X, VIEW_LIST_Y = VIEWS[args.views]
    print(f"view convention = {args.views}: x={VIEW_LIST_X} y={VIEW_LIST_Y}")

    if not os.path.exists(args.data_path):
        print(f"Error: data path '{args.data_path}' does not exist!")
        raise SystemExit(1)

    cls_list = list(POINT_SIZE.keys())
    if args.only_class:
        if args.only_class not in POINT_SIZE:
            print(f"Error: unknown class '{args.only_class}'. "
                  f"Valid: {sorted(POINT_SIZE)}")
            raise SystemExit(1)
        cls_list = [args.only_class]

    for cls in cls_list:
        point_size = POINT_SIZE[cls]
        for folder in ["test"]:
            directory_path = os.path.join(args.data_path, cls, folder)
            if not os.path.exists(directory_path):
                print(f"\nWarning: not found, skipping: {directory_path}")
                continue
            print(f"Processing class '{cls}' (point_size={point_size}), folder '{folder}'")
            process_directory(directory_path, point_size, args.out_dirname,
                              args.light_on, args.limit)
