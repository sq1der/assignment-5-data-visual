import os
import sys
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import math

def mkdirp(p):
    os.makedirs(p, exist_ok=True)

def clamp(x, a, b):
    return max(a, min(b, x))

def estimate_parameters(bbox_diag):
    """
    Возвращает подобранные параметры на основе диагонали bounding box (в тех же единицах, что модель).
    Правила подобраны эмпирически:
      - более крупная модель -> больше точек и большая глубина Poisson
      - voxel_size ~ diag / k
      - radius для нормалей ~ diag / 50..150
    """
    # diag в метрах/единицах модели
    diag = float(bbox_diag)
    # Число точек для семплинга (если исходный объект - mesh)
    # базовый множитель: чем больше диагональ — тем больше точек. Огр. сверху 300k, снизу 8k
    point_count = int(clamp(diag * 5000, 50000, 120000))
    # Voxel size: хотим ~200..10000 вокселей по диагонали, так что voxel = diag / N
    voxel_size = float(clamp(diag / 200.0, 0.0005, diag / 30.0))
    # Радиус поиска нормалей: от diag/200 до diag/30
    normal_radius = float(clamp(diag / 80.0, diag / 200.0, diag / 30.0))
    # Poisson depth: 8..12 (больше — медленнее, детальнее)
    if diag < 0.2:
        poisson_depth = 7
    elif diag < 1.0:
        poisson_depth = 8
    elif diag < 3.0:
        poisson_depth = 9
    else:
        poisson_depth = 9
    # плотность порога при очистке Poisson: квантиль
    density_quantile = 0.01
    # сфера для отметки экстремумов: радиус ~ diag/100
    marker_radius = float(clamp(diag / 100.0, 0.001, diag / 20.0))

    return {
        "point_count": point_count,
        "voxel_size": voxel_size,
        "normal_radius": normal_radius,
        "poisson_depth": poisson_depth,
        "density_quantile": density_quantile,
        "marker_radius": marker_radius
    }

# ----- Загрузка модели (mesh или pointcloud) -----
def load_input(path: str):
    ext = Path(path).suffix.lower()
    if ext in ['.ply', '.pcd', '.xyz', '.xyzn', '.pts']:
        # try point cloud first
        try:
            pcd = o3d.io.read_point_cloud(path)
            if not pcd.is_empty():
                return "pcd", pcd
        except Exception:
            pass
    # try mesh
    try:
        mesh = o3d.io.read_triangle_mesh(path)
        if not mesh.is_empty():
            return "mesh", mesh
    except Exception:
        pass
    # fallback: try read_point_cloud again (some .ply are point clouds)
    try:
        pcd = o3d.io.read_point_cloud(path)
        if not pcd.is_empty():
            return "pcd", pcd
    except Exception:
        pass
    raise RuntimeError(f"Не удалось загрузить файл как mesh или pointcloud: {path}")

# ----- Статистика -----
def compute_stats_geom(geom):
    if isinstance(geom, o3d.geometry.PointCloud):
        pts = np.asarray(geom.points)
        n = pts.shape[0]
        bbox = geom.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        extent = bbox.get_extent()
        has_colors = geom.has_colors()
        return {"type":"pcd","n_points":n,"bbox":bbox,"diag":diag,"extent":extent,"has_colors":has_colors}
    elif isinstance(geom, o3d.geometry.TriangleMesh):
        v = np.asarray(geom.vertices)
        t = np.asarray(geom.triangles)
        n_v = v.shape[0]
        n_t = t.shape[0]
        bbox = geom.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        extent = bbox.get_extent()
        has_colors = geom.has_vertex_colors()
        has_normals = geom.has_vertex_normals()
        return {"type":"mesh","n_vertices":n_v,"n_triangles":n_t,"bbox":bbox,"diag":diag,"extent":extent,"has_colors":has_colors,"has_normals":has_normals}
    else:
        raise RuntimeError("Unknown geometry type for stats")

# ----- Основные шаги (реализация похожая на ранее) -----
def step1_load(path, out_dir):
    kind, geom = load_input(path)
    stats = compute_stats_geom(geom)
    print("Loaded as:", kind)
    for k,v in stats.items():
        if k=='bbox' or k=='extent': continue
        print(f"  {k}: {v}")
    # Сохраним исходник в outputs/ для проверки
    if kind == "pcd":
        o3d.io.write_point_cloud(os.path.join(out_dir,"input_pcd.ply"), geom)
    else:
        o3d.io.write_triangle_mesh(os.path.join(out_dir,"input_mesh.ply"), geom)
    # визуализируем
    o3d.visualization.draw_geometries([geom], window_name="Step1: Original")
    return kind, geom, stats

def step2_to_pcd(kind, geom, params, out_dir):
    # Возвращаем point cloud и сохраняем
    if kind == "pcd":
        pcd = geom
    else:
        # sampling from mesh
        n = params["point_count"]
        print(f"Sampling {n} points from mesh...")
        pcd = geom.sample_points_uniformly(number_of_points=n)
    print("PointCloud points:", np.asarray(pcd.points).shape[0])
    o3d.io.write_point_cloud(os.path.join(out_dir,"step2_pcd.ply"), pcd)
    o3d.visualization.draw_geometries([pcd], window_name="Step2: PointCloud")
    return pcd

def step3_poisson(pcd, params, out_dir):
    print("Estimating normals with radius:", params["normal_radius"])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=params["normal_radius"], max_nn=30))
    pcd.normalize_normals()
    print("Running Poisson (depth=%d)..." % params["poisson_depth"])
    mesh_raw, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=params["poisson_depth"])
    densities = np.asarray(densities)
    # remove low-density vertices
    thresh = np.quantile(densities, params["density_quantile"])
    print("Poisson density threshold (quantile %.3f): %.6f" % (params["density_quantile"], thresh))
    verts_to_keep = densities > thresh
    mesh_crop = mesh_raw
    # try crop by pointcloud bbox to remove far away artifacts
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_crop = mesh_raw.crop(bbox)
    print("Poisson mesh (cropped) vertices:", np.asarray(mesh_crop.vertices).shape[0], "triangles:", np.asarray(mesh_crop.triangles).shape[0])
    o3d.io.write_triangle_mesh(os.path.join(out_dir,"step3_poisson.ply"), mesh_crop)
    o3d.visualization.draw_geometries([mesh_crop], window_name="Step3: Poisson")
    return mesh_crop

def step4_voxelize(pcd, params, out_dir):
    print("Creating VoxelGrid with voxel_size:", params["voxel_size"])
    vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=params["voxel_size"])
    print("Voxel count:", len(vox.get_voxels()))
    # VoxelGrid нельзя прямо экспортировать в ply; сохраним как json-ish npz (coords)
    vox_coords = [v.grid_index for v in vox.get_voxels()]
    np.savez_compressed(os.path.join(out_dir,"step4_voxels.npz"), coords=np.array(vox_coords))
    # для визуализации: создадим small cubes for a subset (если много вокселей — берём первые 5000)
    mesh_voxels = []
    max_show = 5000
    for i, v in enumerate(vox.get_voxels()):
        if i >= max_show: break
        cube = o3d.geometry.TriangleMesh.create_box(width=params["voxel_size"], height=params["voxel_size"], depth=params["voxel_size"])
        center = (np.array(v.grid_index) + vox.origin) * params["voxel_size"]
        cube.translate(center)
        cube.compute_vertex_normals()
        mesh_voxels.append(cube)
    if mesh_voxels:
        o3d.visualization.draw_geometries(mesh_voxels, window_name="Step4: Voxels (subset)")
    return vox

def step5_add_plane(geom, params, out_dir):
    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    # создаём плоскость размером примерно равным наибольшему измерению * 1.2
    size = max(extent) * 1.2
    plane = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size*0.001)
    plane.compute_vertex_normals()
    # разместим плоскость сбоку (по положительному X)
    plane_center = center - np.array([0, 0, extent[2] * 0.15])
    plane.translate(plane_center - np.array([size / 2, size / 2, 0]))
    plane_normal = np.array([0.0, 0.0, 1.0])
    plane.paint_uniform_color([0.8,0.8,0.8])
    o3d.io.write_triangle_mesh(os.path.join(out_dir,"step5_plane.ply"), plane)
    o3d.visualization.draw_geometries([geom, plane], window_name="Step5: Plane")
    # нормаль плоскости направлена вдоль +X в этом расположении
    return plane, plane_center, np.array([1.0,0.0,0.0])

def step6_clip(pcd, plane_point, plane_normal, out_dir):
    pts = np.asarray(pcd.points)
    n = np.array(plane_normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    dots = (pts - plane_point).dot(n)
    mask = dots <= 0.0
    pts_remaining = pts[mask]
    pcd_clipped = o3d.geometry.PointCloud()
    pcd_clipped.points = o3d.utility.Vector3dVector(pts_remaining)
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)[mask]
        pcd_clipped.colors = o3d.utility.Vector3dVector(cols)
    print("Points remaining after clipping:", pts_remaining.shape[0])
    o3d.io.write_point_cloud(os.path.join(out_dir,"step6_clipped.ply"), pcd_clipped)
    o3d.visualization.draw_geometries([pcd_clipped], window_name="Step6: Clipped")
    return pcd_clipped

def step7_color_and_extrema(pcd, params, axis='x', out_dir=None):
    pts = np.asarray(pcd.points)
    axis_idx = {'x':0,'y':1,'z':2}[axis]
    vals = pts[:, axis_idx]
    vmin, vmax = vals.min(), vals.max()
    norm = (vals - vmin) / (vmax - vmin + 1e-12)
    colors = np.vstack([norm, np.zeros_like(norm), 1-norm]).T
    pcd.colors = o3d.utility.Vector3dVector(colors)
    min_idx = np.argmin(vals)
    max_idx = np.argmax(vals)
    min_pt = pts[min_idx]
    max_pt = pts[max_idx]
    # маркеры
    r = params["marker_radius"]
    s_min = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    s_min.paint_uniform_color([1,0,0])
    s_min.translate(min_pt)
    s_max = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    s_max.paint_uniform_color([0,1,0])
    s_max.translate(max_pt)
    o3d.io.write_point_cloud(os.path.join(out_dir,"step7_colored.ply"), pcd)
    o3d.visualization.draw_geometries([pcd, s_min, s_max], window_name="Step7: Gradient + Extrema")
    print(f"Extrema {axis}: min = {min_pt}, max = {max_pt}")
    return min_pt, max_pt

# ----- Основной рабочий процесс -----
def process_all(input_path, output_dir):
    mkdirp(output_dir)
    kind, geom, stats = step1_load(input_path, output_dir)
    diag = stats["diag"]
    params = estimate_parameters(diag)
    print("\nАвтоподобранные параметры (на основе диагонали bbox = %.6f):" % diag)
    for k,v in params.items():
        print(f"  {k}: {v}")
    # шаг 2
    pcd = step2_to_pcd(kind, geom, params, output_dir)
    # шаг 3
    mesh_rec = step3_poisson(pcd, params, output_dir)
    # шаг 4
    vox = step4_voxelize(pcd, params, output_dir)
    # шаг 5
    plane, plane_center, plane_normal = step5_add_plane(geom if kind=="mesh" else pcd, params, output_dir)
    # шаг 6
    pcd_clipped = step6_clip(pcd, plane_center, plane_normal, output_dir)
    # шаг 7
    extrema = step7_color_and_extrema(pcd_clipped, params, axis='x', out_dir=output_dir)
    print("\nГотово — результаты сохранены в:", output_dir)
    return output_dir, params, extrema

# ----- CLI -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Путь к 3D файлу (.ply/.obj/.stl/.pcd и т.д.)")
    ap.add_argument("--out", "-o", default="outputs", help="Папка для результатов")
    args = ap.parse_args()
    input_path = args.input
    out_dir = args.out
    if not os.path.exists(input_path):
        print("Файл не найден:", input_path)
        sys.exit(1)
    process_all(input_path, out_dir)

if __name__ == "__main__":
    main()
