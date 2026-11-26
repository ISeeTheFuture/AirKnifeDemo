"""Load the five supplied captures. Distances in all modules are metres."""

import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy import ndimage

# Manually selected pixel regions (left, top, right, bottom), not product labels.
CAPTURES = {
    1: ("172321019", (145, 113, 484, 299), 0.009),
    2: ("174713171", (205, 175, 407, 266), 0.008),
    3: ("174804559", (227, 130, 391, 301), 0.008),
    4: ("174854756", (192, 157, 363, 364), 0.006),
    5: ("174947708", (262, 182, 334, 246), 0.010),
}


@dataclass
class Scan:
    sample: int
    xyz: np.ndarray
    valid: np.ndarray
    selected: np.ndarray
    height: np.ndarray
    roi: tuple
    components: int
    center: np.ndarray
    size: np.ndarray
    angle: float
    axis_ratio: float
    surface: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray


def locate_tar():
    bundled = Path(__file__).with_name("Air Knife System Samples.tar")
    if bundled.is_file():
        return bundled
    for folder in (
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path.home() / "Downloads",
    ):
        example = folder / "Air Knife System 검토 의뢰.tar"
        if example.is_file():
            return example.resolve()
        candidates = [p for p in folder.glob("Air Knife System*.tar") if p.is_file()]
        if len(candidates) == 1:
            return candidates[0].resolve()
    return None


def load_scan(filename, sample):
    capture, (x0, y0, x1, y1), cutoff = CAPTURES[sample]
    prefix = f"02 Proposal/{sample}/PCL_{capture}/PCL_"
    axes = []
    with tarfile.open(filename, "r:*", encoding="utf-8") as src:
        members = {m.name.removeprefix("./"): m for m in src if m.isfile()}
        for axis in "XYZ":
            name = prefix + axis + "_0.csv"
            if name not in members:
                raise ValueError(f"촬영 파일을 찾을 수 없습니다. {name}")
            if members[name].size > 8_000_000:
                raise ValueError(f"지원하는 촬영 파일 크기를 넘었습니다. {name}")
            with src.extractfile(members[name]) as data:
                arr = np.loadtxt(BytesIO(data.read()), delimiter=",")
            if arr.shape != (480, 640):
                raise ValueError(f"{name}의 크기가 480×640과 다릅니다.")
            axes.append(arr)
    xyz = np.stack(axes, axis=-1)
    xx, yy, zz = axes
    valid = np.isfinite(xyz).all(axis=2) & (zz > 0)
    roi = np.zeros_like(valid)
    roi[y0:y1, x0:x1] = True
    ring = np.zeros_like(valid)
    ring[max(0, y0 - 25) : y1 + 25, max(0, x0 - 25) : x1 + 25] = True
    ring &= valid & ~roi
    design = np.stack([xx, yy, np.ones_like(zz)], axis=-1)
    inliers = ring.copy()
    for _ in range(5):
        if inliers.sum() < 100:
            raise ValueError("배경 평면을 추정할 점이 부족합니다.")
        coef = np.linalg.lstsq(design[inliers], zz[inliers], rcond=None)[0]
        height = design @ coef - zz
        inliers = ring & (np.abs(height) < 0.003)
    candidates = roi & valid & (height >= cutoff)
    labels, _ = ndimage.label(
        ndimage.binary_closing(candidates, iterations=2), structure=np.ones((3, 3))
    )
    counts = np.bincount(labels[candidates], minlength=1)
    keep = np.flatnonzero(counts >= 35)
    keep = keep[keep != 0]
    mask = candidates & np.isin(labels, keep)
    if mask.sum() < 80:
        raise ValueError("관심 영역에서 제품 후보를 분리하지 못했습니다.")
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[mask]))
    mean, covariance = cloud.compute_mean_and_covariance()
    points = xyz[mask, :2]
    center = np.asarray(mean)[:2]
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(covariance)[:2, :2])
    major = eigenvectors[:, -1]
    size = np.ptp((points - center) @ eigenvectors[:, ::-1], axis=0)
    angle = np.rad2deg(np.arctan2(major[1], major[0])) % 180
    # Mock jig: sensor X -> rack horizontal; negative sensor Y -> rack vertical.
    # Each row here is horizontal offset, vertical offset, protrusion.
    local = np.stack((xx - center[0], center[1] - yy, height), axis=-1)
    remote = np.zeros_like(mask)
    remote[::3, ::3] = True
    surface = local[mask & remote]
    # Triangulate only measured, connected pixels. Holes remain unfilled.
    stride = 2
    grid = local[::stride, ::stride]
    active = mask[::stride, ::stride]
    index = np.full(active.shape, -1, dtype=np.int32)
    index[active] = np.arange(active.sum())
    a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    triangles = np.concatenate(
        (np.stack((a, b, c), -1).reshape(-1, 3), np.stack((b, d, c), -1).reshape(-1, 3))
    )
    triangles = triangles[(triangles >= 0).all(1)]
    if not len(triangles) or not len(surface):
        raise ValueError("표면을 구성할 연결점이 부족합니다.")
    return Scan(
        sample,
        xyz,
        valid,
        mask,
        height,
        (x0, y0, x1, y1),
        len(keep),
        center,
        size,
        angle,
        eigenvalues[1] / max(eigenvalues[0], 1e-12),
        surface,
        grid[active],
        triangles,
    )


def open_cloud(scan):
    import open3d as o3d

    # Use the same mask as the planner. Only downsample the displayed background.
    xyz = scan.xyz[scan.valid]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pc.paint_uniform_color([0.24, 0.35, 0.43])
    pc = pc.voxel_down_sample(0.003)
    product = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(scan.xyz[scan.selected])
    )
    product.paint_uniform_color([1, 0.63, 0.19])
    o3d.visualization.draw_geometries(
        [pc, product],
        window_name=f"ToF capture {scan.sample} | orange: selected",
        width=960,
        height=680,
        front=[0, 0, -1],
        up=[0, -1, 0],
        lookat=np.median(xyz, axis=0),
        zoom=0.74,
    )
