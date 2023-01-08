import json
import os

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from pprint import PrettyPrinter
from pyworkforce.utils.shift_spec import get_shift_coverage, decode_shift_spec, unwrap_shift
from pyworkforce.plotters.matplotlib import plot
import pandas as pd
import numpy as np
from pyworkforce.plotters.scheduling import plot, plot_xy_per_interval

scheduler_data_path = os.path.dirname(os.path.realpath(__file__))

# with open(os.path.join(scheduler_data_path, '../scheduling_output1.json'), 'r') as f:
#     solution = json.load(f)

# resources_shifts = solution['resources_shifts']
# df = pd.DataFrame(resources_shifts)

# df['shifted_resources_per_slot'] = df.apply(lambda t: np.array(unwrap_shift(t['shift'])) * t['resources'], axis=1)

# # df['shifted_resources_per_slot'].to_csv('test.csv', ',', index=False)

# df1 = df[['day', 'shifted_resources_per_slot']].groupby('day', as_index=False)['shifted_resources_per_slot'].apply(lambda x: np.sum(np.vstack(x), axis = 0)).to_frame()#.reset_index()

# np.set_printoptions(linewidth=np.inf, formatter=dict(float=lambda x: "%3.0i" % x))

# df1.to_csv('shifted_resources_per_slot.csv')
# arr = df1['shifted_resources_per_slot'].values
# arr = np.concatenate(arr)

# df3 = pd.read_csv('../scheduling_output_stage1.csv')

# df3['resources_shifts'] = arr.tolist()
# print(df3)

# df3.to_csv('../scheduling_output_stage2.csv')

df3 = pd.read_csv('../scheduling_output_stage2.csv')

plot_xy_per_interval("positions_per_interval.png", df3, x="tc", y=["positions", "resources_shifts"])