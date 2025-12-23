#!/usr/bin/env python3
"""
End-to-end Neuralangelo-style pipeline:
1) Extract frames from a video at a target FPS.
2) Run COLMAP (via Neuralangelo repo scripts) to estimate poses.
3) Convert COLMAP -> transforms.json.
4) Generate a Neuralangelo training config.
5) Train.
6) Extract a high-resolution mesh (marching cubes over learned SDF).

Key robustness fix:
- Some Neuralangelo COLMAP scripts expect images in datasets/<seq>/images_2.
  This driver always writes to images_2 and also creates a compatible images alias
  (symlink or directory fallback), so either expectation works.

Assumptions:
- Run from Neuralangelo repo root (or pass --repo_root).
- Installed: ffmpeg, colmap, torchrun, and Neuralangelo deps.
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import re
import shutil
import subprocess
import sys
import torch
from typing import Optional


def shlex_quote(s: str) -> str:
    """Minimal shlex-like quoting for printing commands."""
    if re.fullmatch(r"[A-Za-z0-9_/\-.:=]+", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _run_cmd(
    cmd: list[str],
    cwd: Optional[pathlib.Path] = None,
    env: Optional[dict[str, str]] = None,
) -> None:
    """Runs a command and raises on failure."""
    cmd_str = " ".join([shlex_quote(x) for x in cmd])
    print(f"\n==> {cmd_str}")
    try:
        subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed (exit {e.returncode}): {cmd_str}") from e


def _require_exists(path: pathlib.Path, what: str) -> None:
    """Ensures a filesystem path exists."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def _which_or_raise(tool: str) -> None:
    """Ensures an external tool exists on PATH."""
    if shutil.which(tool) is None:
        raise FileNotFoundError(
            f"Required tool not found on PATH: {tool}\nInstall it and try again."
        )


def _normalize_scene_type(scene_type: str) -> str:
    """Normalizes scene type spellings."""
    s = scene_type.strip().lower()
    if s in ("indoor", "indoors"):
        return "indoor"
    if s in ("outdoor", "outdoors"):
        return "outdoor"
    if s in ("object", "objects"):
        return "object"
    raise ValueError(
        f"Unsupported --scene_type={scene_type!r}. Use: indoor | outdoor | object."
    )


def _pick_latest_checkpoint(log_dir: pathlib.Path) -> pathlib.Path:
    """Picks the newest .pt checkpoint file under a log directory."""
    candidates: list[pathlib.Path] = []
    for p in log_dir.rglob("*.pt"):
        if p.is_file():
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints (*.pt) found under: {log_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _best_effort_symlink(link_path: pathlib.Path, target: pathlib.Path) -> bool:
    """Best-effort symlink creation. Returns True if link exists after call."""
    try:
        if link_path.is_symlink() or link_path.exists():
            return True
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(target, target_is_directory=True)
        return True
    except OSError:
        return link_path.exists()


def _ensure_colmap_image_dir_aliases(
    dataset_dir: pathlib.Path, primary_images_dirname: str
) -> pathlib.Path:
    """
    Ensures dataset/images, dataset/images_2, and dataset/images_raw exist as valid
    directories/aliases.

    Some Neuralangelo COLMAP scripts use images_2; others use images; run_colmap.sh
    uses images_raw. This makes all three paths resolve to the extracted frames.

    Returns the directory path we should write frames into.
    """
    images = dataset_dir / "images"
    images_2 = dataset_dir / "images_2"
    images_raw = dataset_dir / "images_raw"

    # Step 1: Decide which directory will physically hold the frames.
    if primary_images_dirname == "images":
        primary = images
    elif primary_images_dirname == "images_2":
        primary = images_2
    else:
        raise ValueError("primary_images_dirname must be 'images' or 'images_2'.")

    # Step 2: Create primary directory.
    primary.mkdir(parents=True, exist_ok=True)

    # Step 3: Create all other variants as symlinks to primary.
    for alias_path in [images, images_2, images_raw]:
        if alias_path == primary or alias_path.exists():
            continue
        linked = _best_effort_symlink(alias_path, primary)
        if not linked:
            alias_path.mkdir(parents=True, exist_ok=True)

    return primary


def _extract_frames_ffmpeg(
    video_path: pathlib.Path, images_dir: pathlib.Path, fps: float
) -> None:
    """Extracts frames as PNGs at a fixed FPS."""
    # Step 1: Ensure required tools exist.
    _which_or_raise("ffmpeg")

    # Step 2: Extract frames at desired FPS with high fidelity PNG output.
    images_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = images_dir / "frame_%06d.png"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(out_pattern),
    ]
    _run_cmd(cmd)

    # Step 3: Validate extraction.
    num_frames = len(list(images_dir.glob("frame_*.png")))
    if num_frames < 10:
        raise RuntimeError(
            f"Too few frames extracted ({num_frames}). "
            f"Try lowering --fps or using a longer video segment."
        )
    print(f"Extracted {num_frames} frames into: {images_dir}")


def _main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--video_path", required=True, type=str, help="Path to input video file."
    )
    parser.add_argument(
        "--fps",
        default=3.0,
        type=float,
        help="Frame sampling rate in frames per second.",
    )
    parser.add_argument(
        "--scene_type",
        default="indoors",
        type=str,
        help="Scene type: indoor|outdoor|object.",
    )
    parser.add_argument(
        "--resolution",
        default=2048,
        type=int,
        help="Mesh extraction resolution (higher = more detail).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--repo_root",
        default=".",
        type=str,
        help="Neuralangelo repo root (contains train.py, projects/...).",
    )
    parser.add_argument(
        "--gpus", default=torch.cuda.device_count(), type=int, help="GPUs for torchrun --nproc_per_node."
    )
    parser.add_argument(
        "--group",
        default="na_runs",
        type=str,
        help="Experiment group name (log folder grouping).",
    )
    parser.add_argument(
        "--name",
        default="",
        type=str,
        help="Run name override. Default: derived from timestamp.",
    )
    parser.add_argument(
        "--block_res",
        default=128,
        type=int,
        help="Extraction chunk size; lower if OOM.",
    )
    parser.add_argument(
        "--auto_exposure_wb",
        action="store_true",
        help="Enable auto exposure/white-balance modeling if supported.",
    )
    parser.add_argument(
        "--primary_images_dir",
        default="images_2",
        choices=["images", "images_2"],
        help="Where to physically write extracted frames; "
        "the other directory will be created as an alias if possible.",
    )
    parser.add_argument(
        "--reuse_frames",
        action="store_true",
        help="If set, do not re-extract frames if they already exist.",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training step (assumes checkpoint already exists).",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        help="Explicit checkpoint path. If empty, newest *.pt in logdir is used.",
    )
    parser.add_argument(
        "--colmap_script",
        default="",
        type=str,
        help="Optional path to run_colmap.sh. If empty, uses repo default.",
    )

    args = parser.parse_args()

    # Step 1: Normalize and validate args.
    repo_root = pathlib.Path(args.repo_root).resolve()
    video_path = pathlib.Path(args.video_path).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    scene_type = _normalize_scene_type(args.scene_type)

    _require_exists(video_path, "video file")
    _require_exists(repo_root / "train.py", "Neuralangelo train.py (repo root)")
    _require_exists(
        repo_root / "projects" / "neuralangelo" / "scripts",
        "Neuralangelo scripts directory",
    )

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.resolution < 64:
        raise ValueError("--resolution looks too small; use >= 64")
    if args.gpus < 1:
        raise ValueError("--gpus must be >= 1")

    _which_or_raise("colmap")

    # Step 2: Prepare run naming + directories.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name.strip() or f"mesh_{timestamp}"
    run_root = output_dir / run_name
    dataset_dir = run_root / "dataset"
    logs_root = run_root / "logs"
    mesh_dir = run_root / "mesh"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo root:   {repo_root}")
    print(f"Run root:    {run_root}")
    print(f"Scene type:  {scene_type}")
    print(f"FPS:         {args.fps}")
    print(f"Resolution:  {args.resolution}")

    # Step 3: Ensure images + images_2 aliases exist and choose where to write frames.
    images_dir = _ensure_colmap_image_dir_aliases(
        dataset_dir=dataset_dir,
        primary_images_dirname=args.primary_images_dir,
    )

    # Step 4: Extract frames.
    existing = sorted(images_dir.glob("frame_*.png"))
    if args.reuse_frames and existing:
        print(f"Reusing {len(existing)} existing frames in: {images_dir}")
    else:
        _extract_frames_ffmpeg(
            video_path=video_path, images_dir=images_dir, fps=args.fps
        )

    # Step 5: Some repos assume ./datasets/<sequence>. Make a best-effort symlink.
    _best_effort_symlink(repo_root / "datasets" / run_name, dataset_dir)

    # Step 6: Run COLMAP via Neuralangelo helper script.
    if args.colmap_script:
        run_colmap_sh = pathlib.Path(args.colmap_script).expanduser().resolve()
    else:
        run_colmap_sh = (
            repo_root / "projects" / "neuralangelo" / "scripts" / "run_colmap.sh"
        )
    _require_exists(run_colmap_sh, "run_colmap.sh")

    # Step 6a: Clean partial COLMAP outputs if sparse is empty or missing.
    sparse_dir = dataset_dir / "sparse"
    if sparse_dir.exists() and not any(sparse_dir.rglob("*")):
        shutil.rmtree(sparse_dir, ignore_errors=True)
    # (Optional) keep database.db if you want incremental runs; default is clean slate.
    if (dataset_dir / "database.db").exists():
        (dataset_dir / "database.db").unlink()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    _run_cmd(["bash", str(run_colmap_sh), str(dataset_dir)], cwd=repo_root, env=env)

    # Step 6b: Verify COLMAP produced a model.
    model0 = dataset_dir / "sparse" / "0"
    if not model0.exists():
        raise RuntimeError(
            "COLMAP did not produce sparse/0. "
            "If you still see ExistsDir(*image_path), your run_colmap.sh is using an "
            "unexpected images folder name. This driver creates BOTH images and images_2, "
            "so that usually indicates the script is pointing somewhere else."
        )

    # Step 7: Convert COLMAP outputs to transforms.json.
    convert_py = (
        repo_root / "projects" / "neuralangelo" / "scripts" / "convert_data_to_json.py"
    )
    _require_exists(convert_py, "convert_data_to_json.py")

    convert_cmd = [
        sys.executable,
        str(convert_py),
        "--data_dir",
        str(dataset_dir),
        "--scene_type",
        scene_type,
    ]
    if args.auto_exposure_wb:
        convert_cmd.append("--auto_exposure_wb")
    _run_cmd(convert_cmd, cwd=repo_root, env=env)

    transforms_json = dataset_dir / "transforms.json"
    _require_exists(transforms_json, "transforms.json")

    # Step 8: Generate a training config.
    generate_py = (
        repo_root / "projects" / "neuralangelo" / "scripts" / "generate_config.py"
    )
    _require_exists(generate_py, "generate_config.py")

    _run_cmd(
        [
            sys.executable,
            str(generate_py),
            "--sequence_name",
            run_name,
            "--data_dir",
            str(dataset_dir),
            "--scene_type",
            scene_type,
        ],
        cwd=repo_root,
        env=env,
    )

    cfg_custom = (
        repo_root
        / "projects"
        / "neuralangelo"
        / "configs"
        / "custom"
        / f"{run_name}.yaml"
    )
    _require_exists(cfg_custom, "generated config yaml")

    # Step 9: Train.
    logdir = logs_root / args.group / run_name
    logdir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        train_cmd = [
            "torchrun",
            f"--nproc_per_node={args.gpus}",
            "train.py",
            f"--logdir={logdir}",
            f"--config={cfg_custom}",
            "--show_pbar",
        ]
        _run_cmd(train_cmd, cwd=repo_root, env=env)

    # Step 10: Choose checkpoint.
    if args.checkpoint.strip():
        ckpt = pathlib.Path(args.checkpoint).expanduser().resolve()
        _require_exists(ckpt, "checkpoint")
    else:
        ckpt = _pick_latest_checkpoint(logdir)
    print(f"Using checkpoint: {ckpt}")

    # Step 11: Extract mesh.
    extract_py = repo_root / "projects" / "neuralangelo" / "scripts" / "extract_mesh.py"
    _require_exists(extract_py, "extract_mesh.py")

    log_config = logdir / "config.yaml"
    config_for_extract = log_config if log_config.exists() else cfg_custom

    out_mesh = mesh_dir / f"{run_name}_res{args.resolution}.ply"
    extract_cmd = [
        "torchrun",
        f"--nproc_per_node={args.gpus}",
        str(extract_py),
        "--config",
        str(config_for_extract),
        "--checkpoint",
        str(ckpt),
        "--output_file",
        str(out_mesh),
        "--resolution",
        str(args.resolution),
        "--block_res",
        str(args.block_res),
        "--keep_lcc",
        "--textured",
    ]
    _run_cmd(extract_cmd, cwd=repo_root, env=env)

    _require_exists(out_mesh, "output mesh file")
    print("\n✅ Done.")
    print(f"Mesh: {out_mesh}")
    print(f"Run root: {run_root}")


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
