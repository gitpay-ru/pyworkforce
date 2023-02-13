from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from pyworkforce.solver_params import SolverParams

import pandas as pd
import json

from pyworkforce.staffing import MultiZonePlanner
import pytz
eastern = pytz.timezone('US/Eastern')


input_csv_path = '../data_file.csv'
input_meta_path = '../meta_file.json'
output_dir = '../out2'

Path(output_dir).mkdir(parents=True, exist_ok=True)
df = pd.read_csv(input_csv_path, parse_dates=[0], index_col=0)

with open(input_meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)

solver_params = {
    'schedule': SolverParams(do_logging=True, max_iteration_search_time=float(5*60), num_search_workers=16),
    'roster': SolverParams(do_logging=True, max_iteration_search_time=float(120*60), num_search_workers=16),
    'roster_breaks': SolverParams(do_logging=True, max_iteration_search_time=float(60*60), num_search_workers=16),
}

mzp = MultiZonePlanner(df, meta, output_dir, solver_params)
mzp.solve()

# mzp.schedule()
# mzp.roster()
# mzp.roster_breaks()
# mzp.roster_postprocess()
# mzp.combine_results()
# mzp.recalculate_stats()

