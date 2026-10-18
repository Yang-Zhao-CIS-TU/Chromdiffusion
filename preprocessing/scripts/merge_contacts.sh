#!/bin/bash
# Merge the per-replicate merged_nodups files of one cell line into a single contact list,
# sorted the way Juicer `pre` requires: chr1, then chr2, then pos1, then pos2.
#
# Usage: merge_contacts.sh <input_dir> <output_file> [sort_tmp_dir]
#   input_dir     folder with the *_merged_nodups.txt.gz files from raw_data_download_script.sh
#   output_file   e.g. <cell_test_dir>/juicer_ready/GM12878/total_merged_nodups.txt
#   sort_tmp_dir  temporary space for sort; needs roughly the size of the output (default: $TMPDIR or /tmp)
#
# Columns in these files: readname str1 chr1 pos1 frag1 str2 chr2 pos2 frag2 mapq1 mapq2,
# so the sort keys are 3 (chr1), 7 (chr2), 4 (pos1) and 8 (pos2).

set -euo pipefail

IN_DIR=${1:?"Usage: merge_contacts.sh <input_dir> <output_file> [sort_tmp_dir]"}
OUT=${2:?"Provide the output file"}
SORT_TMP=${3:-${TMPDIR:-/tmp}}

shopt -s nullglob
files=("${IN_DIR}"/*_merged_nodups.txt.gz)
if [[ ${#files[@]} -eq 0 ]]; then
    echo "[ERROR] No *_merged_nodups.txt.gz files in ${IN_DIR}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT}")"
echo "[INFO] Merging ${#files[@]} files into ${OUT}"
gzip -dc "${files[@]}" | LC_ALL=C sort -k3,3V -k7,7V -k4,4n -k8,8n -T "${SORT_TMP}" > "${OUT}"
echo "[INFO] Done: $(wc -l < "${OUT}") contacts"
