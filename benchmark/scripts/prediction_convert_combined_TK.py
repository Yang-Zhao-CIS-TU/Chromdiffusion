#!/usr/bin/env python3
"""
================================================================================
COMBINED PREDICTION CONVERSION SCRIPT
================================================================================

Converts model prediction tiles back into Juicer-compatible contact matrices
for downstream feature calling (Arrowhead, HiCCUPS).

TWO CONVERSION STAGES (combined):
  1. NPZ → NPY      - Load model output, transpose channel-first to channel-last
  2. NPY → TXT      - Map tiles to genome coordinates, average overlaps, output Juicer format

Input formats supported:
  - .npz files (ChromDiff output): hic_impute_chr{N}_pred_real.npz
  - .npy files (pre-converted):    preds_lr_test_chr{N}_ratio{R}.npy

Output:
  - preds_lr_test_chr{N}_ratio{R}.txt         (sparse matrix: pos1, pos2, score)
  - preds_lr_test_chr{N}_ratio{R}_convert.txt (Juicer BEDPE format for `juicer pre`)

Usage:
    # From NPZ files (full pipeline)
    python prediction_convert_combined_TK.py --input-dir <npz_folder> --input-format npz

    # From NPY files (skip transpose)
    python prediction_convert_combined_TK.py --input-dir <npy_folder> --input-format npy

Author: Combined from convert_npz_to_npy.py, convert_predictions_to_hic_input_TK.py
================================================================================
"""

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np


# ==============================================================================
# SECTION 1: DEFAULT PARAMETERS
# ==============================================================================

DEFAULT_CHROMS = [f"chr{i}" for i in range(18, 23)]
DEFAULT_RATIO = 16
DEFAULT_RESOLUTION = 10_000
DEFAULT_CHUNK_SIZE = 40
DEFAULT_THRESH_BINS = 200
DEFAULT_MIN_NONZERO_FRAC = 0.05


# ==============================================================================
# SECTION 2: ARGUMENT PARSING
# ==============================================================================

def parse_args() -> argparse.Namespace:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    workspace_root = os.path.join(repo_root, "GM12878_test")
    default_juicer_root = os.path.join(workspace_root, "juicer_ready")
    default_chrom_sizes = os.path.join(workspace_root, "metadata", "hg19.chrom.sizes")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/output arguments
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing prediction files (.npz or .npy)"
    )
    parser.add_argument(
        "--input-format", choices=["npz", "npy"], default="npz",
        help="Input format: 'npz' (model output) or 'npy' (pre-converted) (default: npz)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: same as input-dir)"
    )

    # Reference data arguments
    parser.add_argument(
        "--juicer-root", default=default_juicer_root,
        help=f"Root with HR/LR intra_NONE dumps (default: {default_juicer_root})"
    )
    parser.add_argument(
        "--hr-cell", default="GM12878",
        help="HR folder name under juicer_root (default: %(default)s)"
    )
    parser.add_argument(
        "--lr-cell", default=None,
        help="LR folder name under juicer_root (default: <hr-cell>_LR_rep)"
    )
    parser.add_argument(
        "--chrom-sizes", default=default_chrom_sizes,
        help=f"Chromosome sizes file (default: {default_chrom_sizes})"
    )

    # Processing parameters
    parser.add_argument(
        "--chroms", nargs="+", default=DEFAULT_CHROMS,
        help="Chromosomes to process (default: chr18-22)"
    )
    parser.add_argument(
        "--ratio", type=str, default=DEFAULT_RATIO,
        help="Downsample ratio (default: %(default)s)"
    )
    parser.add_argument(
        "--resolution", type=int, default=DEFAULT_RESOLUTION,
        help="Bin size in bp (default: %(default)s)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Tile size used during training (default: %(default)s)"
    )
    parser.add_argument(
        "--thresh-bins", type=int, default=DEFAULT_THRESH_BINS,
        help="Max |i-j| (in bins) to keep a tile (default: %(default)s)"
    )
    parser.add_argument(
        "--min-nonzero-frac", type=float, default=DEFAULT_MIN_NONZERO_FRAC,
        help="Minimum nonzero fraction in LR tile to keep it (default: %(default)s)"
    )

    # Output options
    parser.add_argument(
        "--strip-chr-prefix", action="store_true",
        help="Drop 'chr' prefix in output (useful for mm9)"
    )
    parser.add_argument(
        "--save-npy", action="store_true",
        help="Also save intermediate .npy files (only relevant for npz input)"
    )

    return parser.parse_args()


# ==============================================================================
# SECTION 3: CHROMOSOME SIZE LOADING
# ==============================================================================

def load_chrom_sizes(path: str) -> Dict[str, int]:
    """
    Load chromosome sizes from a tab-separated file.

    Args:
        path: Path to chrom.sizes file (chr<N> <length>)

    Returns:
        Dict mapping chromosome names to lengths (both with and without 'chr' prefix)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Chromosome sizes file not found: {path}")

    sizes: Dict[str, int] = {}
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
            # Store both with and without chr prefix for flexible lookup
            core = name[3:] if name.startswith("chr") else name
            sizes[core] = length
            sizes[f"chr{core}"] = length

    if not sizes:
        raise ValueError(f"No chromosome lengths could be parsed from {path}")
    return sizes


# ==============================================================================
# SECTION 4: LR CONTACT MATRIX LOADING
# ==============================================================================

def load_contact_matrix(file_path: str, mat_dim: int, res: int) -> np.ndarray:
    """
    Load sparse contact matrix from Juicer dump text file.

    Args:
        file_path: Path to intra_NONE text file
        mat_dim: Matrix dimension (bins)
        res: Resolution in bp

    Returns:
        Symmetric contact matrix as 2D numpy array
    """
    mat = np.zeros((mat_dim, mat_dim), dtype=np.float32)
    with open(file_path) as fh:
        for line in fh:
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            idx1 = int(float(parts[0]))
            idx2 = int(float(parts[1]))
            val = float(parts[2])
            if idx1 // res >= mat_dim or idx2 // res >= mat_dim:
                continue
            mat[idx1 // res, idx2 // res] = val
    # Make symmetric
    mat = mat + mat.T - np.diag(mat.diagonal())
    return mat


def load_lr_contacts(
    juicer_root: str,
    lr_cell: str,
    chroms: List[str],
    res: int,
    ratio: int,
    chrom_sizes: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """
    Load LR contact matrices for all chromosomes.

    These are needed to regenerate tile indices (matching the training preprocessing).
    """
    res_label = f"{res // 1000}k"
    lr_dir = os.path.join(juicer_root, lr_cell, "intra_NONE")
    contacts: Dict[str, np.ndarray] = {}

    for chrom in chroms:
        chrom_len = chrom_sizes.get(chrom) or chrom_sizes.get(chrom.lstrip("chr"))
        if chrom_len is None:
            raise KeyError(f"Missing chromosome length for {chrom} in sizes file.")

        mat_dim = int(math.ceil(chrom_len / res))
        lr_path = os.path.join(
            lr_dir, f"{chrom}_{res_label}_intra_NONE_downsample_ratio{ratio}.txt"
        )

        if not os.path.isfile(lr_path):
            raise FileNotFoundError(f"LR intra file not found: {lr_path}")

        contacts[chrom] = load_contact_matrix(lr_path, mat_dim, res)
        print(f"  Loaded LR matrix: {chrom} ({mat_dim}x{mat_dim})")

    return contacts


# ==============================================================================
# SECTION 5: TILE INDEX GENERATION
# ==============================================================================

def generate_tile_indices(
    lr_mat: np.ndarray,
    chunk_size: int,
    thresh_bins: int,
    min_nonzero_frac: float
) -> List[Tuple[int, int]]:
    """
    Generate tile indices matching the training preprocessing.

    CRITICAL: This must exactly mirror NPYtensorBuilderTKcustom.py logic
    to ensure tile indices align with model predictions.

    Args:
        lr_mat: LR contact matrix
        chunk_size: Tile size (default: 40)
        thresh_bins: Max diagonal distance in bins (default: 200)
        min_nonzero_frac: QC threshold for minimum nonzero entries

    Returns:
        List of (row_idx, col_idx) tuples for valid tiles
    """
    rows, cols = lr_mat.shape
    if rows <= thresh_bins or cols <= thresh_bins:
        raise ValueError(f"Hi-C matrix too small for chunking: {rows}x{cols}")

    def passes_qc(tile: np.ndarray) -> bool:
        return np.count_nonzero(tile) >= min_nonzero_frac * tile.size

    indices: List[Tuple[int, int]] = []
    for idx1 in range(0, rows - chunk_size, chunk_size):
        for idx2 in range(0, cols - chunk_size, chunk_size):
            # Only keep tiles near diagonal (within thresh_bins)
            if abs(idx1 - idx2) < thresh_bins:
                lr_tile = lr_mat[idx1:idx1 + chunk_size, idx2:idx2 + chunk_size]
                if passes_qc(lr_tile):
                    indices.append((idx1, idx2))

    if not indices:
        raise ValueError("No tiles passed QC; check thresholds or inputs.")

    return indices


# ==============================================================================
# SECTION 6: NPZ/NPY LOADING AND CONVERSION
# ==============================================================================

def load_predictions_npz(npz_path: str) -> np.ndarray:
    """
    Load predictions from NPZ file (ChromDiff output format).

    Input shape:  (N, 1, 40, 40) - channel-first
    Output shape: (N, 40, 40, 1) - channel-last
    """
    data = np.load(npz_path)["Y_pred"]  # Shape: (N, 1, 40, 40)
    # Transpose channel-first to channel-last
    data = np.transpose(data, (0, 2, 3, 1))  # Shape: (N, 40, 40, 1)
    return data


def load_predictions_npy(npy_path: str, chunk_size: int) -> np.ndarray:
    """
    Load predictions from NPY file (pre-converted format).

    Handles both channel-last and channel-first formats.
    """
    pred_tiles = np.load(npy_path)

    # Ensure channel-last format
    if pred_tiles.ndim == 3:
        pred_tiles = pred_tiles[..., np.newaxis]
    elif pred_tiles.ndim == 4 and pred_tiles.shape[1] == 1:
        # Channel-first with singleton channel: transpose
        pred_tiles = np.transpose(pred_tiles, (0, 2, 3, 1))
    elif pred_tiles.ndim != 4:
        raise ValueError(f"Expected 3D or 4D predictions, got shape {pred_tiles.shape}")

    if pred_tiles.shape[1] != chunk_size or pred_tiles.shape[2] != chunk_size:
        raise ValueError(
            f"Prediction tiles have unexpected size {pred_tiles.shape[1:3]}, "
            f"expected {chunk_size}x{chunk_size}."
        )

    return pred_tiles


# ==============================================================================
# SECTION 7: TILE TO CONTACT CONVERSION
# ==============================================================================

def tiles_to_entries(
    pred_tiles: np.ndarray,
    indices: List[Tuple[int, int]],
    res: int
) -> List[Tuple[int, int, float]]:
    """
    Convert prediction tiles back to genomic contact entries.

    AVERAGING LOGIC:
    Multiple tiles can overlap at the same genomic position.
    We accumulate scores and counts, then compute the average.

    Args:
        pred_tiles: Prediction array (N, 40, 40, 1)
        indices: Tile index positions from generate_tile_indices
        res: Resolution in bp

    Returns:
        List of (pos1, pos2, score) tuples, sorted by position
    """
    if pred_tiles.shape[0] != len(indices):
        raise ValueError(
            f"Prediction count ({pred_tiles.shape[0]}) != tile indices ({len(indices)})"
        )

    # Accumulate scores for averaging overlapping tiles
    accumulated: Dict[Tuple[int, int], List[float]] = {}

    for tile, (idx1, idx2) in zip(pred_tiles, indices):
        tile_2d = tile[:, :, 0]
        nz_rows, nz_cols = np.nonzero(tile_2d)

        for r, c in zip(nz_rows, nz_cols):
            pos1 = (idx1 + r) * res
            pos2 = (idx2 + c) * res
            # Ensure consistent ordering (upper triangle)
            key = (pos1, pos2) if pos1 <= pos2 else (pos2, pos1)

            if key not in accumulated:
                accumulated[key] = [tile_2d[r, c], 1.0]
            else:
                accumulated[key][0] += tile_2d[r, c]
                accumulated[key][1] += 1.0

    # Compute averages
    averaged = [
        (p1, p2, score_sum / count)
        for (p1, p2), (score_sum, count) in accumulated.items()
    ]
    averaged.sort(key=lambda x: (x[0], x[1]))

    return averaged


# ==============================================================================
# SECTION 8: OUTPUT WRITING
# ==============================================================================

def write_sparse_txt(entries: List[Tuple[int, int, float]], output_path: str) -> None:
    """Write sparse matrix format (pos1, pos2, score)."""
    with open(output_path, "w") as fh:
        for pos1, pos2, score in entries:
            fh.write(f"{pos1}\t{pos2}\t{score}\n")


def write_juicer_bedpe(
    entries: List[Tuple[int, int, float]],
    chrom: str,
    output_path: str,
    strip_chr_prefix: bool
) -> None:
    """
    Write Juicer BEDPE format for `juicer_tools pre`.

    Format: 0 chr pos1 0 0 chr pos2 1 score
    """
    chrom_name = chrom[3:] if strip_chr_prefix and chrom.startswith("chr") else chrom
    with open(output_path, "w") as fh:
        for pos1, pos2, score in entries:
            fh.write(f"0 {chrom_name} {pos1} 0 0 {chrom_name} {pos2} 1 {score}\n")


# ==============================================================================
# SECTION 9: MAIN EXECUTION
# ==============================================================================

def main() -> None:
    args = parse_args()
    lr_cell = args.lr_cell or f"{args.hr_cell}_LR_rep"
    output_dir = args.output_dir or args.input_dir

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("COMBINED PREDICTION CONVERSION")
    print("=" * 70)
    print(f"Input dir:    {args.input_dir}")
    print(f"Input format: {args.input_format}")
    print(f"Output dir:   {output_dir}")
    print(f"Chromosomes:  {args.chroms}")
    print(f"Ratio:        {args.ratio}")
    print("=" * 70)

    # Load chromosome sizes
    print("\n[1/4] Loading chromosome sizes...")
    chrom_sizes = load_chrom_sizes(args.chrom_sizes)

    # Load LR contact matrices (needed to regenerate tile indices)
    print("\n[2/4] Loading LR contact matrices...")
    lr_contacts = load_lr_contacts(
        args.juicer_root, lr_cell, args.chroms,
        args.resolution, args.ratio, chrom_sizes
    )

    # Generate tile indices for each chromosome
    print("\n[3/4] Generating tile indices...")
    tile_index_map = {
        chrom: generate_tile_indices(
            lr_contacts[chrom],
            args.chunk_size,
            args.thresh_bins,
            args.min_nonzero_frac
        )
        for chrom in args.chroms
    }
    for chrom, indices in tile_index_map.items():
        print(f"  {chrom}: {len(indices)} tiles")

    # Process each chromosome
    print("\n[4/4] Converting predictions...")
    for chrom in args.chroms:
        print(f"\n  Processing {chrom}...")

        # Determine input file path based on format
        if args.input_format == "npz":
            # Look for ChromDiff output naming pattern
            input_path = os.path.join(
                args.input_dir,
                f"hic_impute_{chrom}_pred_real.npz"
            )
            if not os.path.isfile(input_path):
                # Try alternative naming
                chrom_num = chrom.replace("chr", "")
                input_path = os.path.join(
                    args.input_dir,
                    f"hic_impute_chr{chrom_num}_pred_real.npz"
                )

            if not os.path.isfile(input_path):
                print(f"    Warning: NPZ file not found for {chrom}, skipping")
                continue

            print(f"    Loading NPZ: {input_path}")
            pred_tiles = load_predictions_npz(input_path)

            # Optionally save intermediate NPY
            if args.save_npy:
                npy_path = os.path.join(
                    output_dir,
                    f"preds_lr_test_{chrom}_ratio{args.ratio}.npy"
                )
                np.save(npy_path, pred_tiles)
                print(f"    Saved NPY: {npy_path}")

        else:  # npy format
            input_path = os.path.join(
                args.input_dir,
                f"preds_lr_test_{chrom}_ratio{args.ratio}.npy"
            )
            if not os.path.isfile(input_path):
                print(f"    Warning: NPY file not found for {chrom}, skipping")
                continue

            print(f"    Loading NPY: {input_path}")
            pred_tiles = load_predictions_npy(input_path, args.chunk_size)

        print(f"    Predictions shape: {pred_tiles.shape}")

        # Validate tile count matches
        expected_tiles = len(tile_index_map[chrom])
        if pred_tiles.shape[0] != expected_tiles:
            print(f"    ERROR: Prediction count ({pred_tiles.shape[0]}) != "
                  f"expected tiles ({expected_tiles})")
            continue

        # Convert tiles to contact entries
        entries = tiles_to_entries(pred_tiles, tile_index_map[chrom], args.resolution)
        print(f"    Generated {len(entries)} contact entries")

        # Write output files
        base_name = f"preds_lr_test_{chrom}_ratio{args.ratio}"

        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        write_sparse_txt(entries, txt_path)

        convert_path = os.path.join(output_dir, f"{base_name}_convert.txt")
        write_juicer_bedpe(entries, chrom, convert_path, args.strip_chr_prefix)

        print(f"    Wrote: {txt_path}")
        print(f"    Wrote: {convert_path}")

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Output location: {output_dir}")
    print("\nGenerated files:")
    print("  - *_ratio{R}.txt         (sparse matrix format)")
    print("  - *_ratio{R}_convert.txt (Juicer BEDPE format)")
    print("\nNext step: Run feature_calling_combined_TK.sh")


if __name__ == "__main__":
    main()
