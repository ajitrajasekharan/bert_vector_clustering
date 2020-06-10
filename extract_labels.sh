#cat debug_pivots.txt |  awk '{if ($1 != "_singletons_" && $1 != "_empty_") print $1,$2,$3,$5,$6}' > tmp_labels.txt
cat debug_pivots.txt |  awk '{if ($1 != "_singletons_" && $1 != "_empty_") print $3,$5,$6}' > tmp_labels.txt
cat map_labels.txt | awk '{print $1,$2}' > user_labels.txt
paste -d ' ' user_labels.txt tmp_labels.txt > labels.txt
python label_stats.py
cat labels.txt | awk '{if (NF != 5) print $0}'


