import argparse
import os

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

# must match the settings used to generate 2d_rendering / 2d_3d_cor
IMG_SIZE = 336
WINDOW = 1036
POINT_SIZE = 10

# angles are pre-compensated for the base rotation matrix_x(0.25*pi) (+45 deg); net x rotation = 45 + v_x. Do not change.
VIEW_LIST_X = [-90, -60, -45, -30, -0]
VIEW_LIST_Y = [-30, -15, 15, 30]

CLS_LIST = ["Marshmallow", "PeppermintCandy"]


def resize_organized_pc(organized_pc, target_height=IMG_SIZE, target_width=IMG_SIZE,
                        tensor_out=False, mode="nearest"):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).float()
    torch_resized = torch.nn.functional.interpolate(
        torch_organized_pc, size=(target_height, target_width), mode=mode)
    if tensor_out:
        return torch_resized.squeeze(dim=0)
    return torch_resized.squeeze(dim=0).permute(1, 2, 0).numpy()


def matrix_x(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[1, 0, 0, 0],
                     [0, cos_angle, -sin_angle, 0],
                     [0, sin_angle, cos_angle, 0],
                     [0, 0, 0, 1]])


def fit_camera(vis, o3d_pc_r, ref_xy, screen_width, screen_height, max_iter=200):
    """Step the camera along z (0.005 per step) and keep the step whose projection matches
    ref_xy on the most pixels. Returns (k, exact_count, n).

    cor is the gather index used by back_to_3d, so it must match the render pixel for pixel.
    Without ref_xy, use the first position where all points are on screen.
    """
    pts = np.asarray(o3d_pc_r.points)
    n = len(pts)
    view_control = vis.get_view_control()

    records = []      # (k, extrinsic, exact, on_screen)
    for k in range(max_iter + 1):
        if k > 0:
            cam = view_control.convert_to_pinhole_camera_parameters()
            ext = np.array(cam.extrinsic)
            ext[:3, 3] = ext[0:3, 3] + np.array([0, 0, 0.005])
            cam.extrinsic = ext
            view_control.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

        param = view_control.convert_to_pinhole_camera_parameters()
        K = param.intrinsic.intrinsic_matrix
        ext = np.array(param.extrinsic)
        r = cv2.Rodrigues(ext[:3, :3])[0]
        t = ext[:3, 3:]
        p2d, _ = cv2.projectPoints(pts, r, t, K, None)
        xs, ys = p2d[:, 0, 0], p2d[:, 0, 1]
        on_screen = not (np.any(xs < 0) or np.any(xs >= screen_width) or
                         np.any(ys < 0) or np.any(ys >= screen_height))

        exact = -1
        if ref_xy is not None:
            mine = (p2d[:, 0, :] * IMG_SIZE / WINDOW).astype(int).clip(0, IMG_SIZE - 1)
            exact = int(np.all(mine == ref_xy, axis=1).sum())

        records.append((k, ext, exact, on_screen))
        if exact == n:                                  # perfect match, stop
            break
        if ref_xy is None and on_screen and k > 0:      # no reference: fall back to the on-screen criterion
            break

    if ref_xy is not None:
        best = max(records, key=lambda rec: rec[2])
    else:
        ons = [rec for rec in records if rec[3] and rec[0] > 0]
        best = ons[0] if ons else records[-1]

    cam = view_control.convert_to_pinhole_camera_parameters()
    cam.extrinsic = best[1]
    view_control.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    return best[0], best[2], n


def project_points(vis, o3d_pc_r):
    """Recompute (unrounded) pixel coordinates to compare with the existing 2d_3d_cor."""
    control = vis.get_view_control()
    param = control.convert_to_pinhole_camera_parameters()
    intrinsic = param.intrinsic.intrinsic_matrix
    extrinsic = param.extrinsic
    r = cv2.Rodrigues(extrinsic[:3, :3])[0]
    t = extrinsic[:3, 3:]
    points_2d, _ = cv2.projectPoints(np.asarray(o3d_pc_r.points), r, t, intrinsic, None)
    return points_2d[:, 0, :] * IMG_SIZE / WINDOW


def render_color(o3d_pc_r, view_idx, color_save_path, point_size, light_on, ref_cor_dir):
    """Render one view. Returns a per-view diagnostics dict, or None without a reference cor."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=WINDOW, height=WINDOW, visible=False)
    vis.add_geometry(o3d_pc_r)

    ref_xy = None
    n_ref = None
    if ref_cor_dir is not None:
        ref_path = os.path.join(ref_cor_dir, f"view_{view_idx}_cor.npy")
        if os.path.exists(ref_path):
            ref = np.load(ref_path)
            n_ref = len(ref)
            if n_ref == len(o3d_pc_r.points):
                ref_xy = ref[:, :2].astype(int)
            else:
                print(f"      !! view {view_idx}: cor 行数={n_ref} "
                      f"本地点数={len(o3d_pc_r.points)}，跳过自检")

    k, exact_cnt, n = fit_camera(vis, o3d_pc_r, ref_xy, WINDOW, WINDOW)

    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([1, 1, 1])
    # lighting off by default to get pure albedo, complementing the gray shaded renders
    render_option.light_on = light_on
    render_option.point_size = point_size
    render_option.mesh_show_back_face = True

    stat = None
    if ref_xy is not None:
        mine_f = project_points(vis, o3d_pc_r)
        d = mine_f - ref_xy.astype(np.float64)
        stat = {
            "view": view_idx,
            "n": n,
            "k": k,
            "exact": exact_cnt / n,
            "dx": float(np.median(d[:, 0])),
            "dy": float(np.median(d[:, 1])),
            "maxerr": float(np.percentile(np.abs(d).max(axis=1), 99)),
            "within1": float(np.mean(np.all(np.abs(d) <= 1.0, axis=1))),
        }

    image = vis.capture_screen_float_buffer(do_render=True)
    img = Image.fromarray((np.asarray(image) * 255).astype(np.uint8))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    img.save(f"{color_save_path}/view_{view_idx}.png")
    vis.destroy_window()
    return stat


def get_mv_color_images(pcd_path, rgb_path, cor_dir, color_save_path, point_size, light_on):
    o3d_pc = o3d.io.read_point_cloud(pcd_path)
    if not o3d_pc.has_points():
        print(f"    Warning: no points in {pcd_path}, skip.")
        return
    n_pts = len(o3d_pc.points)

    # align colors with the point cloud using the same kept_index
    kept_index = np.load(os.path.join(cor_dir, "nonzero_indices.npy"))
    if len(kept_index) != n_pts:
        print(f"    !! 跳过：pcd 点数={n_pts} 与 nonzero_indices 长度={len(kept_index)} 不符")
        return

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float64) / 255.0
    rgb = resize_organized_pc(rgb, IMG_SIZE, IMG_SIZE).reshape(-1, 3)
    colors = rgb[kept_index, :]
    colors = np.clip(colors.astype(np.float64), 0.0, 1.0)  # Vector3dVector needs float64

    o3d_pc.colors = o3d.utility.Vector3dVector(colors)
    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    o3d_pc.transform(matrix_x(0.25 * np.pi))

    checks = []
    view_idx = 0
    for v_x in VIEW_LIST_X:
        o3d_pc_o = o3d.geometry.PointCloud(o3d_pc)
        R = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(v_x), 0, 0])
        o3d_pc_r = o3d_pc_o.rotate(R, center=o3d_pc.get_center())
        c = render_color(o3d_pc_r, view_idx, color_save_path, point_size, light_on, cor_dir)
        if c:
            checks.append(c)
        view_idx += 1
    for v_y in VIEW_LIST_Y:
        o3d_pc_o = o3d.geometry.PointCloud(o3d_pc)
        # undo the base +45 deg so net x is zero, then apply a pure R_y
        R_x = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(-45), 0, 0])
        o3d_pc_o.rotate(R_x, center=o3d_pc.get_center())
        R_y = o3d_pc.get_rotation_matrix_from_axis_angle([0, np.radians(v_y), 0])
        o3d_pc_o.rotate(R_y, center=o3d_pc.get_center())
        c = render_color(o3d_pc_o, view_idx, color_save_path, point_size, light_on, cor_dir)
        if c:
            checks.append(c)
        view_idx += 1

    if checks:
        # exact must be 100%; dx/dy/max help tell a scale mismatch from a wrong angle
        for c in checks:
            print(f"      view {c['view']}: k={c['k']:3d}  exact={c['exact']*100:5.1f}%  "
                  f"<=1px={c['within1']*100:5.1f}%  dx={c['dx']:+.1f} dy={c['dy']:+.1f}  "
                  f"max={c['maxerr']:.1f}")
        exact = sum(c["exact"] * c["n"] for c in checks) / sum(c["n"] for c in checks)
        w1 = sum(c["within1"] * c["n"] for c in checks) / sum(c["n"] for c in checks)
        n_perfect = sum(1 for c in checks if c["exact"] >= 1.0)
        ks = sorted({c["k"] for c in checks})
        if n_perfect == len(checks):
            tag = f"全部视角逐像素命中 cor，k={ks}"
        elif exact > 0.999:
            tag = "基本命中"
        else:
            bad = [c["view"] for c in checks if c["exact"] < 1.0]
            tag = (f"{n_perfect}/{len(checks)} 视角命中；view {bad} 搜不到匹配的 k "
                   f"=> 差异不是相机距离，需另查")
        print(f"    [cor check] exact={exact*100:.2f}%  <=1px={w1*100:.2f}%  => {tag}")


def process_condition_dir(cond_dir, out_dirname, light_on, limit):
    """cond_dir = <cls>/test/{good,anomaly}, containing pcd/ rgb/ 2d_3d_cor/."""
    pcd_dir = os.path.join(cond_dir, "pcd")
    rgb_dir = os.path.join(cond_dir, "rgb")
    cor_root = os.path.join(cond_dir, "2d_3d_cor")
    if not os.path.isdir(pcd_dir):
        print(f"  skip, no pcd/: {cond_dir}")
        return 0

    done = 0
    for file in sorted(os.listdir(pcd_dir)):
        if not file.endswith(".pcd"):
            continue
        if limit and done >= limit:
            break
        file_id = os.path.splitext(file)[0]
        rgb_path = os.path.join(rgb_dir, f"{file_id}.png")
        cor_dir = os.path.join(cor_root, file_id)
        if not os.path.exists(rgb_path):
            print(f"  !! 缺 rgb，跳过：{rgb_path}")
            continue
        if not os.path.exists(os.path.join(cor_dir, "nonzero_indices.npy")):
            print(f"  !! 缺 nonzero_indices.npy，跳过：{cor_dir}")
            continue

        color_save_path = os.path.join(cond_dir, out_dirname, file_id)
        os.makedirs(color_save_path, exist_ok=True)
        print(f"  {os.path.join(pcd_dir, file)}")
        get_mv_color_images(os.path.join(pcd_dir, file), rgb_path, cor_dir,
                            color_save_path, POINT_SIZE, light_on)
        done += 1
    return done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate multi-view true-color images for Eyecandies", add_help=True)
    parser.add_argument("--data_path", type=str,
                        default="/root/autodl-tmp/Eyecandies_processed",
                        help="处理后的 Eyecandies 根目录，含 <cls>/test/{good,anomaly}/{pcd,rgb,2d_3d_cor}")
    parser.add_argument("--out_dirname", type=str, default="2d_color",
                        help="输出目录名，与 2d_depth / 2d_rendering 同级")
    parser.add_argument("--only_class", type=str, default=None,
                        help="只处理某一个类，用于先做校验")
    parser.add_argument("--limit", type=int, default=0,
                        help="每个 good/anomaly 目录最多处理几个样本，0 表示不限")
    parser.add_argument("--light_on", action="store_true",
                        help="打开光照。默认关闭以得到纯 albedo")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: data path '{args.data_path}' does not exist!")
        raise SystemExit(1)

    cls_list = CLS_LIST
    if args.only_class:
        if args.only_class not in CLS_LIST:
            print(f"Error: unknown class '{args.only_class}'. Valid: {CLS_LIST}")
            raise SystemExit(1)
        cls_list = [args.only_class]

    for cls in cls_list:
        for subfolder in ["test"]:
            for condition in ["good", "anomaly"]:
                cond_dir = os.path.join(args.data_path, cls, subfolder, condition)
                if not os.path.isdir(cond_dir):
                    print(f"Info: not found, skip: {cond_dir}")
                    continue
                print(f"Processing {cls}/{subfolder}/{condition} (point_size={POINT_SIZE})")
                process_condition_dir(cond_dir, args.out_dirname, args.light_on, args.limit)
