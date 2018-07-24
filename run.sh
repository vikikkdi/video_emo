echo $1
scenedetect --input $1 --detector threshold --threshold 16 --min-percent 90 --csv-output sceneout.csv
python csv_reader.py $1
