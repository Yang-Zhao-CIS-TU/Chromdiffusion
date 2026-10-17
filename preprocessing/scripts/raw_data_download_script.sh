#!/bin/bash
#Usage: bash raw_data_download_script.sh <cell_type> [output_dir]
#  Downloads the per-replicate *_merged_nodups.txt.gz files of one cell type from GEO (GSE63525)
#  into output_dir (default: ./<cell_type>).
# the code is provided: https://github.com/kimmo1019/hicGAN (via DFHiC). Changed here: output
# directory argument, HTTPS instead of FTP, GSM number = HIC index + 549, CH12-LX removed.

#Cell type indexes are listed as follows (refer to GSE63525):
#GM12878 idx: 1-18
#K562 idx:69-74
#IMR90 idx:50-56
#NHEK idx:65-67
declare -i start_idx
declare -i end_idx
declare -i idx

DPATH=${2:-"$1"}
mkdir -p "$DPATH"
	
if [ $1 == 'GM12878' ]
then
	start_idx=1
	end_idx=18
elif [ $1 == 'K562' ]
then
	start_idx=69
	end_idx=74
elif [ $1 == 'IMR90' ]
then
	start_idx=50
	end_idx=56
elif [ $1 == 'NHEK' ]
then
	start_idx=65
	end_idx=67
else
	echo "The input cell type is not identified, please refer to GEO database with accession number GSE63525"
	exit 1
fi

for ((idx=$start_idx; idx<=$end_idx; idx++))
do
	number=$(expr $idx + 549)
	wget -c -P "$DPATH" https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551`printf "%03d" $number`/suppl/GSM1551`printf "%03d" $number`_HIC`printf "%03d" $idx`_merged_nodups.txt.gz
done
