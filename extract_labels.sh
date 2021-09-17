cp labels.txt results
echo "Stats in clusters"
python label_stats.py adaptive_debug_pivots.txt
echo "Stats after aggregation"
python label_stats.py labels.txt
cp stats_dict.txt results
#cat labels.txt | awk '{if (NF != 5) print $0}'


