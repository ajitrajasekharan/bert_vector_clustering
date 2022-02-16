python mag_stats.py
python mag_stats.py | tail -n +2   | awk -F'\t' '{if ($4 != 0) print $0}'| sort -k4,4 -n -r  > pivot_entities.tsv
