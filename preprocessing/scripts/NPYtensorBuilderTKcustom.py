#!/usr/bin/env python3
"""
Build HR/LR training and test tensors (NumPy arrays) from Juicer dump outputs.

This is adapted from the DFHiC preprocessing snippet, but wired to the
ExperimentsTK layout produced by `preprocess_rep_TK_25.sh`:

* HR inputs:   <workspace_root>/<hr_cell>/intra_NONE/chr<chrom>_<res_label>_intra_NONE.txt
* LR inputs:   <workspace_root>/<lr_cell>/intra_NONE/chr<chrom>_<res_label>_intra_NONE_downsample_ratio<ratio>.txt
* Outputs:     NPZ files under <output_dir> containing train/test HR/LR tiles and distance metadata.

Key differences vs. the DFHiC snippet:
- Uses argparse instead of bare sys.argv and validates paths.
 - Accepts explicit workspace root, cell names, resolution, ratio, and chrom sizes file.
 - Defaults output to GM12878_test/tensors/data/<hr_cell> to stay inside ExperimentsTK.
 - Uses robust chromosome parsing (handles “chr1” or “1” in the sizes file).
"""

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HR/LR NumPy tensors from Juicer dumps for ExperimentsTK.")
    parser.add_argument(
        "--workspace-root",
        default="/data/tomoya/ExperimentsTK/GM12878_test/juicer_ready",
        help="Root containing HR/LR folders with intra_NONE dumps (default: %(default)s)",
    )
    parser.add_argument(
        "--hr-cell",
        default="GM12878",
        help="High-coverage cell folder name under workspace root (default: %(default)s)",
    )
    parser.add_argument(
        "--lr-cell",
        default=None,
        help="Low-coverage/downsampled cell folder name (default: <hr_cell>_LR_rep)",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        default=16,
        help="Downsample ratio label used in LR filenames (default: %(default)s)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=10_000,
        help="Bin size in bp (default: %(default)s)",
    )
    parser.add_argument(
        "--chrom-sizes",
        required=True,
        help="Path to chromosome sizes file (two columns: chrom length). Accepts chr-prefixed or not.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write NPZ outputs (default: <workspace_root>/../tensors/data/<hr_cell>)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=40,
        help="Tile size (bins) extracted per window (default: %(default)s)",
    )
    parser.add_argument(
        "--thresh-bins",
        type=int,
        default=200,
        help="Distance threshold in bins (abs(i-j) < thresh) to keep tiles; 200 bins = 2 Mb at 10 kb (default: %(default)s)",
    )
    parser.add_argument(
        "--min-nonzero-frac",
        type=float,
        default=0.05,
        help="Minimum nonzero fraction in LR tile to keep it (default: %(default)s)",
    )
    return parser.parse_args()


def load_chrom_sizes(path: str) -> Dict[str, int]:
    """Parse chrom sizes file; stores both with and without 'chr' prefixes."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Chromosome sizes file not found: {path}")
    mapping: Dict[str, int] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name, length_str = parts[0], parts[1]
            try:
                length = int(length_str)
            except ValueError:
                continue
            core = name[3:] if name.startswith("chr") else name
            mapping[core] = length
            mapping[f"chr{core}"] = length
    if not mapping:
        raise ValueError(f"No chromosome lengths parsed from {path}")
    return mapping


def load_contact_matrix(file_path: str, mat_dim: int, res: int) -> np.ndarray:
    """Load a Juicer dump (observed NONE) into a symmetric dense matrix."""
    mat = np.zeros((mat_dim, mat_dim), dtype=np.float32)
    with open(file_path) as fh:
        for line in fh:
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            # Cast via float->int to tolerate values formatted as floats in text.
            idx1 = int(float(parts[0]))
            idx2 = int(float(parts[1]))
            val = float(parts[2])
            if idx1 // res >= mat_dim or idx2 // res >= mat_dim:
                continue
            mat[idx1 // res, idx2 // res] = val
    # Symmetrize while preserving the diagonal.
    mat = mat + mat.T - np.diag(mat.diagonal())
    return mat


def build_contact_dicts(
    workspace_root: str,
    hr_cell: str,
    lr_cell: str,
    res: int,
    ratio: int,
    chrom_sizes: Dict[str, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    res_label = f"{res // 1000}k"
    hr_dir = os.path.join(workspace_root, hr_cell, "intra_NONE")
    lr_dir = os.path.join(workspace_root, lr_cell, "intra_NONE")

    hr_contacts: Dict[str, np.ndarray] = {}
    lr_contacts: Dict[str, np.ndarray] = {}
    for chrom in range(1, 23):  # chr1-chr22
        chr_name = f"chr{chrom}"
        chrom_len = chrom_sizes.get(chr_name)
        if chrom_len is None:
            raise KeyError(f"Missing chromosome length for {chr_name} in sizes file")
        mat_dim = int(math.ceil(chrom_len / res))

        hr_file = os.path.join(hr_dir, f"{chr_name}_{res_label}_intra_NONE.txt")
        lr_file = os.path.join(
            lr_dir, f"{chr_name}_{res_label}_intra_NONE_downsample_ratio{ratio}.txt"
        )
        if not os.path.isfile(hr_file):
            raise FileNotFoundError(f"HR intra file not found: {hr_file}")
        if not os.path.isfile(lr_file):
            raise FileNotFoundError(f"LR intra file not found: {lr_file}")

        hr_contacts[chr_name] = load_contact_matrix(hr_file, mat_dim, res)
        lr_contacts[chr_name] = load_contact_matrix(lr_file, mat_dim, res)
    return hr_contacts, lr_contacts


def crop_hic_matrix_by_chrom(
    chrom: str,
    hr_contacts: Dict[str, np.ndarray],
    lr_contacts: Dict[str, np.ndarray],
    chunk_size: int,
    thresh_bins: int,
    min_nonzero_frac: float,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str]], List[Tuple[int, int]], Dict[str, int]]:
    """Tile a chromosome's HR/LR matrices into paired crops near the diagonal.

    Returns:
        crop_mats_hr: HR tiles, shape (N, chunk_size, chunk_size)
        crop_mats_lr: LR tiles, shape (N, chunk_size, chunk_size)
        distances: List of (diagonal_offset, chrom) per tile
        locations: List of (idx1, idx2) tile positions in bins
        stats: Dict with 'matrix_shape', 'considered', 'kept', 'skipped_qc'
    """
    hr_mat = hr_contacts[chrom]
    lr_mat = lr_contacts[chrom]
    rows, cols = hr_mat.shape
    if rows <= thresh_bins or cols <= thresh_bins:
        raise ValueError(f"Hi-C matrix too small for {chrom}: {rows}x{cols}")

    def passes_qc(tile: np.ndarray) -> bool:
        return np.count_nonzero(tile) >= min_nonzero_frac * tile.size

    distances: List[Tuple[int, str]] = []
    locations: List[Tuple[int, int]] = []
    crops_hr: List[np.ndarray] = []
    crops_lr: List[np.ndarray] = []
    considered = 0
    skipped_qc = 0
    for idx1 in range(0, rows - chunk_size, chunk_size):
        for idx2 in range(0, cols - chunk_size, chunk_size):
            if abs(idx1 - idx2) < thresh_bins:
                considered += 1
                lr_tile = lr_mat[idx1 : idx1 + chunk_size, idx2 : idx2 + chunk_size]
                if not passes_qc(lr_tile):
                    skipped_qc += 1
                    continue
                hr_tile = hr_mat[idx1 : idx1 + chunk_size, idx2 : idx2 + chunk_size]
                distances.append((idx1 - idx2, chrom))
                locations.append((idx1, idx2))
                crops_lr.append(lr_tile)
                crops_hr.append(hr_tile)

    if not crops_hr or not crops_lr:
        raise ValueError(f"No crops produced for {chrom} with current thresholds")

    crop_mats_hr = np.concatenate([tile[np.newaxis, :] for tile in crops_hr], axis=0)
    crop_mats_lr = np.concatenate([tile[np.newaxis, :] for tile in crops_lr], axis=0)
    stats = {
        "matrix_shape": (rows, cols),
        "considered": considered,
        "kept": len(crops_hr),
        "skipped_qc": skipped_qc,
    }
    return crop_mats_hr, crop_mats_lr, distances, locations, stats


def to_nhwc(arr: np.ndarray) -> np.ndarray:
    """Convert (N, H, W) to (N, H, W, 1) NHWC format."""
    return arr[..., np.newaxis]


def main() -> None:
    args = parse_args()

    lr_cell = args.lr_cell or f"{args.hr_cell}_LR_rep"
    default_output = os.path.abspath(
        os.path.join(args.workspace_root, "..", "tensors", "data", args.hr_cell)
    )
    base_output_dir = args.output_dir or default_output
    output_dir = os.path.join(base_output_dir, f"{args.chunk_size}x{args.chunk_size}")
    os.makedirs(output_dir, exist_ok=True)

    chrom_sizes = load_chrom_sizes(args.chrom_sizes)
    hr_contacts, lr_contacts = build_contact_dicts(
        args.workspace_root, args.hr_cell, lr_cell, args.resolution, args.ratio, chrom_sizes
    )

    train_chroms = [f"chr{i}" for i in range(1, 18)]  # chr1–chr17
    test_chroms = [f"chr{i}" for i in range(18, 23)]  # chr18–chr22

    # Track stats for report
    all_stats: Dict[str, Dict[str, int]] = {}

    # Process training chromosomes
    train_hr_list: List[np.ndarray] = []
    train_lr_list: List[np.ndarray] = []
    train_dist: List[Tuple[int, str]] = []
    train_loc: List[Tuple[int, int]] = []
    for chrom in train_chroms:
        c_hr, c_lr, dist, loc, stats = crop_hic_matrix_by_chrom(
            chrom, hr_contacts, lr_contacts,
            args.chunk_size, args.thresh_bins, args.min_nonzero_frac,
        )
        train_hr_list.append(c_hr)
        train_lr_list.append(c_lr)
        train_dist.extend(dist)
        train_loc.extend(loc)
        all_stats[chrom] = stats

    train_out = os.path.join(output_dir, f"train_data_raw_ratio{args.ratio}.npz")
    np.savez(
        train_out,
        train_lr=to_nhwc(np.concatenate(train_lr_list, axis=0)),
        train_hr=to_nhwc(np.concatenate(train_hr_list, axis=0)),
        distance=train_dist,
        location=train_loc,
    )
    print(f"[DONE] Saved train tensors to {train_out}")

    # Process test chromosomes: save per-chromosome AND accumulate for combined NPZ
    test_hr_list: List[np.ndarray] = []
    test_lr_list: List[np.ndarray] = []
    test_dist: List[Tuple[int, str]] = []
    test_loc: List[Tuple[int, int]] = []
    for chrom in test_chroms:
        c_hr, c_lr, dist, loc, stats = crop_hic_matrix_by_chrom(
            chrom, hr_contacts, lr_contacts,
            args.chunk_size, args.thresh_bins, args.min_nonzero_frac,
        )
        # Accumulate for combined NPZ
        test_hr_list.append(c_hr)
        test_lr_list.append(c_lr)
        test_dist.extend(dist)
        test_loc.extend(loc)
        all_stats[chrom] = stats

        # Save per-chromosome files
        hr_path = os.path.join(output_dir, f"hr_test_{chrom}.npy")
        lr_path = os.path.join(output_dir, f"lr_test_{chrom}_ratio{args.ratio}.npy")
        dist_path = os.path.join(output_dir, f"distance_test_{chrom}.npy")
        loc_path = os.path.join(output_dir, f"location_test_{chrom}.npy")
        np.save(hr_path, to_nhwc(c_hr))
        np.save(lr_path, to_nhwc(c_lr))
        np.save(dist_path, np.array(dist, dtype=object))
        np.save(loc_path, np.array(loc, dtype=np.int32))
        print(f"[DONE] Saved per-chromosome files for {chrom}")

    # Save combined test NPZ (DFHiC-compatible)
    test_out = os.path.join(output_dir, f"test_data_raw_ratio{args.ratio}.npz")
    np.savez(
        test_out,
        test_lr=to_nhwc(np.concatenate(test_lr_list, axis=0)),
        test_hr=to_nhwc(np.concatenate(test_hr_list, axis=0)),
        distance=test_dist,
        location=test_loc,
    )
    print(f"[DONE] Saved combined test tensors to {test_out}")

    # Write build report
    report_path = os.path.join(output_dir, "build_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("NPY Tensor Builder Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("Parameters:\n")
        f.write(f"  Chunk size:        {args.chunk_size}x{args.chunk_size}\n")
        f.write(f"  Resolution:        {args.resolution} bp\n")
        f.write(f"  Threshold bins:    {args.thresh_bins}\n")
        f.write(f"  Min nonzero frac:  {args.min_nonzero_frac}\n")
        f.write(f"  Downsample ratio:  {args.ratio}\n")
        f.write(f"  HR cell:           {args.hr_cell}\n")
        f.write(f"  LR cell:           {lr_cell}\n\n")

        f.write("Per-Chromosome Statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Chrom':<8} {'Matrix Size':<14} {'Considered':<12} {'Kept':<10} {'Skipped QC':<12}\n")
        f.write("-" * 60 + "\n")

        total_considered = 0
        total_kept = 0
        total_skipped = 0
        for chrom in [f"chr{i}" for i in range(1, 23)]:
            st = all_stats[chrom]
            shape_str = f"{st['matrix_shape'][0]}x{st['matrix_shape'][1]}"
            f.write(f"{chrom:<8} {shape_str:<14} {st['considered']:<12} {st['kept']:<10} {st['skipped_qc']:<12}\n")
            total_considered += st["considered"]
            total_kept += st["kept"]
            total_skipped += st["skipped_qc"]

        f.write("-" * 60 + "\n")
        f.write(f"{'TOTAL':<8} {'':<14} {total_considered:<12} {total_kept:<10} {total_skipped:<12}\n\n")

        train_tiles = sum(all_stats[f"chr{i}"]["kept"] for i in range(1, 18))
        test_tiles = sum(all_stats[f"chr{i}"]["kept"] for i in range(18, 23))
        f.write("Summary:\n")
        f.write(f"  Training tiles (chr1-17):  {train_tiles}\n")
        f.write(f"  Test tiles (chr18-22):     {test_tiles}\n")
        f.write(f"  Total tiles:               {total_kept}\n")
        f.write(f"  QC pass rate:              {100 * total_kept / total_considered:.1f}%\n")

    print(f"[DONE] Wrote build report to {report_path}")


if __name__ == "__main__":
    main()
