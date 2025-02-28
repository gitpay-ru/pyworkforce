from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import pandas as pd
import json
from pyworkforce.staffing import MultiZonePlanner
from pyworkforce.staffing.multi_zone_planner import Statuses
import pytz
eastern = pytz.timezone('US/Eastern')


input_csv_path = '../out_475/_data_file_475.csv'
input_meta_path = '../out_475/_meta_file_475.json'
solver_profile_path = '../out_475/_solver_profile_file_475.json'
output_dir = '../out_475/out'


if output_dir and output_dir != '..':
    Path(output_dir).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_csv_path, parse_dates=[0], index_col=0)

with open(input_meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)

with open(solver_profile_path, 'r', encoding='utf-8') as f:
    profile = json.load(f)

mzp = MultiZonePlanner(df, meta, profile, output_dir)
# mzp.solve()

mzp.calc_required_positions()
mzp.schedule()
# mzp.roster()
# mzp.roster_breaks()
# mzp.roster_postprocess()
# mzp.combine_results()
# mzp.recalculate_stats()

# nohup python planner.py &