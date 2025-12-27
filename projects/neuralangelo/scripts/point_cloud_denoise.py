"""
Point cloud denoising utilities for COLMAP sparse reconstructions.
"""

import numpy as np
from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[2]
import sys
sys.path.append(dir_path.__str__())

from third_party.colmap.scripts.python.read_write_model import (
    read_model, write_model, Point3D
)


def filter_by_track_length(points3D: Dict[int, Point3D], min_track_length: int = 3) -> Dict[int, Point3D]:
    """Filter points by minimum track length (number of observations).

    Args:
        points3D: Dictionary mapping point IDs to Point3D objects
        min_track_length: Minimum number of images that must observe a point

    Returns:
        Filtered dictionary of Point3D objects
    """
    filtered = {
        pid: pt for pid, pt in points3D.items()
        if len(pt.image_ids) >= min_track_length
    }
    print(f"Track length filter (min={min_track_length}): "
          f"{len(points3D)} -> {len(filtered)} points")
    return filtered


def filter_by_reprojection_error(points3D: Dict[int, Point3D], max_error: float = 2.0) -> Dict[int, Point3D]:
    """Filter points by maximum reprojection error (in pixels).

    Args:
        points3D: Dictionary mapping point IDs to Point3D objects
        max_error: Maximum reprojection error threshold in pixels

    Returns:
        Filtered dictionary of Point3D objects
    """
    filtered = {
        pid: pt for pid, pt in points3D.items()
        if pt.error < max_error
    }
    print(f"Reprojection error filter (max={max_error}px): "
          f"{len(points3D)} -> {len(filtered)} points")
    return filtered


def statistical_outlier_removal_open3d(
    points3D: Dict[int, Point3D],
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> Dict[int, Point3D]:
    """Remove statistical outliers using Open3D.

    For each point, compute the mean distance to its k nearest neighbors.
    Points whose mean distance is outside the global mean ± std_ratio * std
    are considered outliers.

    Args:
        points3D: Dictionary mapping point IDs to Point3D objects
        nb_neighbors: Number of nearest neighbors to consider
        std_ratio: Standard deviation multiplier threshold

    Returns:
        Filtered dictionary of Point3D objects
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Warning: Open3D not installed. Skipping statistical outlier removal.")
        print("Install with: pip install open3d")
        return points3D

    if len(points3D) == 0:
        return points3D

    # Extract point coordinates
    xyzs = np.stack([pt.xyz for pt in points3D.values()], axis=0)
    point_ids = list(points3D.keys())

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzs)

    # Apply statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    # Filter points
    filtered_ids = [point_ids[i] for i in ind]
    filtered = {pid: points3D[pid] for pid in filtered_ids}

    print(f"Statistical outlier removal (neighbors={nb_neighbors}, std_ratio={std_ratio}): "
          f"{len(points3D)} -> {len(filtered)} points")

    return filtered


def radius_outlier_removal_open3d(
    points3D: Dict[int, Point3D],
    radius: float,
    min_points: int = 5
) -> Dict[int, Point3D]:
    """Remove outliers based on local point density using Open3D.

    Points with fewer than min_points neighbors within the given radius are removed.

    Args:
        points3D: Dictionary mapping point IDs to Point3D objects
        radius: Search radius for neighbors
        min_points: Minimum number of neighbors required within radius

    Returns:
        Filtered dictionary of Point3D objects
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Warning: Open3D not installed. Skipping radius outlier removal.")
        print("Install with: pip install open3d")
        return points3D

    if len(points3D) == 0:
        return points3D

    # Extract point coordinates
    xyzs = np.stack([pt.xyz for pt in points3D.values()], axis=0)
    point_ids = list(points3D.keys())

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzs)

    # Apply radius outlier removal
    cl, ind = pcd.remove_radius_outlier(
        nb_points=min_points,
        radius=radius
    )

    # Filter points
    filtered_ids = [point_ids[i] for i in ind]
    filtered = {pid: points3D[pid] for pid in filtered_ids}

    print(f"Radius outlier removal (radius={radius}, min_points={min_points}): "
          f"{len(points3D)} -> {len(filtered)} points")

    return filtered


def denoise_point_cloud(
    points3D: Dict[int, Point3D],
    args: Optional[Namespace] = None
) -> Dict[int, Point3D]:
    """Apply denoising pipeline to COLMAP point cloud.

    Applies filters in the following order:
    1. Track length filter
    2. Reprojection error filter
    3. Statistical outlier removal (if Open3D available)
    4. Radius outlier removal (if Open3D available)

    Args:
        points3D: Dictionary mapping point IDs to Point3D objects
        args: Namespace with denoising parameters (all optional)

    Returns:
        Filtered dictionary of Point3D objects
    """
    if args is None:
        args = Namespace(
            min_track_length=0,
            max_reprojection_error=float('inf'),
            statistical_outlier=False,
            statistical_nb_neighbors=20,
            statistical_std_ratio=2.0,
            radius_outlier=False,
            radius_outlier_radius=None,
            radius_outlier_min_points=5,
        )

    filtered = points3D
    original_count = len(points3D)

    # Track length filter
    if args.min_track_length > 0:
        filtered = filter_by_track_length(filtered, args.min_track_length)

    # Reprojection error filter
    if args.max_reprojection_error < float('inf'):
        filtered = filter_by_reprojection_error(filtered, args.max_reprojection_error)

    # Statistical outlier removal
    if args.statistical_outlier:
        filtered = statistical_outlier_removal_open3d(
            filtered,
            nb_neighbors=args.statistical_nb_neighbors,
            std_ratio=args.statistical_std_ratio
        )

    # Radius outlier removal
    if args.radius_outlier and args.radius_outlier_radius is not None:
        filtered = radius_outlier_removal_open3d(
            filtered,
            radius=args.radius_outlier_radius,
            min_points=args.radius_outlier_min_points
        )

    final_count = len(filtered)
    if final_count < original_count:
        print(f"Denoising complete: {original_count} -> {final_count} points "
              f"({100*(1-final_count/original_count):.1f}% removed)")
    else:
        print(f"No points removed by denoising (count: {final_count})")

    return filtered


def save_denoised_model(
    cameras: Dict,
    images: Dict,
    points3D: Dict[int, Point3D],
    output_path: str,
    ext: str = ".bin"
) -> None:
    """Save denoised COLMAP model to disk.

    Args:
        cameras: COLMAP cameras dictionary
        images: COLMAP images dictionary
        points3D: Denoised points3D dictionary
        output_path: Path to output directory
        ext: File extension (.bin or .txt)
    """
    os.makedirs(output_path, exist_ok=True)
    sparse_dir = os.path.join(output_path, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    write_model(cameras, images, points3D, sparse_dir, ext=ext)
    print(f"Saved denoised model to {sparse_dir}")


def get_denoise_args(parser: ArgumentParser = None) -> ArgumentParser:
    """Add denoising arguments to an ArgumentParser.

    Args:
        parser: Existing ArgumentParser to extend (creates new one if None)

    Returns:
        ArgumentParser with denoising arguments added
    """
    if parser is None:
        parser = ArgumentParser()

    denoise_group = parser.add_argument_group("Point Cloud Denoising")

    denoise_group.add_argument(
        "--denoise",
        action="store_true",
        help="Enable point cloud denoising"
    )

    denoise_group.add_argument(
        "--min_track_length",
        type=int,
        default=3,
        help="Minimum number of observations (track length) for a point (default: 3)"
    )

    denoise_group.add_argument(
        "--max_reprojection_error",
        type=float,
        default=2.0,
        help="Maximum reprojection error in pixels (default: 2.0)"
    )

    denoise_group.add_argument(
        "--statistical_outlier",
        action="store_true",
        help="Enable statistical outlier removal (requires Open3D)"
    )

    denoise_group.add_argument(
        "--statistical_nb_neighbors",
        type=int,
        default=20,
        help="Number of neighbors for statistical outlier removal (default: 20)"
    )

    denoise_group.add_argument(
        "--statistical_std_ratio",
        type=float,
        default=2.0,
        help="Standard deviation ratio for statistical outlier removal (default: 2.0)"
    )

    denoise_group.add_argument(
        "--radius_outlier",
        action="store_true",
        help="Enable radius-based outlier removal (requires Open3D)"
    )

    denoise_group.add_argument(
        "--radius_outlier_radius",
        type=float,
        default=None,
        help="Search radius for radius outlier removal (required if --radius_outlier is set)"
    )

    denoise_group.add_argument(
        "--radius_outlier_min_points",
        type=int,
        default=5,
        help="Minimum neighbors in radius (default: 5)"
    )

    denoise_group.add_argument(
        "--save_denoised",
        action="store_true",
        help="Save denoised point cloud to disk"
    )

    denoise_group.add_argument(
        "--denoised_output_path",
        type=str,
        default=None,
        help="Output path for denoised model (default: <data_dir>/denoised)"
    )

    return parser


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Denoise COLMAP sparse point cloud"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing sparse/ folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for denoised model (default: <data_dir>/denoised)"
    )
    parser = get_denoise_args(parser)

    args = parser.parse_args()

    # Read COLMAP model
    print(f"Reading COLMAP model from {args.data_dir}")
    cameras, images, points3D = read_model(os.path.join(args.data_dir, "sparse"), ext=".bin")
    print(f"Loaded {len(points3D)} points")

    # Apply denoising
    filtered_points3D = denoise_point_cloud(points3D, args)

    # Save denoised model
    output_dir = args.output_dir or os.path.join(args.data_dir, "denoised")
    save_denoised_model(cameras, images, filtered_points3D, output_dir, ext=".bin")
