
import json
import datetime
import csv

with open(f'.//_data_file_imp.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)

with open(f'.//_meta_file.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)


shifts_duration = {}
for s in meta['shifts']:
    shifts_duration[s['id']] = datetime.timedelta(minutes= int(s['duration'][:-3]) * 60+ int(s['duration'][-2:]))

print(shifts_duration)