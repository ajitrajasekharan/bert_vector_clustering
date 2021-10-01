cp labels.txt results
echo "Stats in clusters"
python label_stats.py adaptive_debug_pivots.txt | tee stats_in_clusters.txt
echo "Stats after aggregation"
python label_stats.py labels.txt | tee stats_dict_with_counts.txt
cp stats_dict.txt results
cat inferred.txt | sort -k3,3 -n -r  > tmp$$
mv tmp$$ inferred.txt
#cat labels.txt | awk '{if (NF != 5) print $0}'
echo "Number of unique types/subtypes in original bootstrap file"
wc -l stats_dict.txt
echo "Number of unique types/subtypes in expanded file"
wc -l stats_dict.txt


