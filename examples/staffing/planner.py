from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import pandas as pd
import json
import argparse
from pyworkforce.staffing import MultiZonePlanner

# input_csv_path = '../scheduling_input.csv'
# input_meta_path = '../scheduling_meta_input.json'
# output_dir = '../out'

# mzp = MultiZonePlanner(df, meta, output_dir)
# mzp.solve()

# mzp.schedule()
# mzp.roster()
# mzp.roster_breaks()
# mzp.roster_postprocess()
# mzp.combine_results()
# mzp.recalculate_stats()

def parse_args():
    parser = argparse.ArgumentParser(description='Process csv and meta.')
    parser.add_argument('-i', '--input', default='../scheduling_input.csv')
    parser.add_argument('-m', '--meta', default='../scheduling_meta_input.json')
    parser.add_argument('-o', '--out', default='../out')
    parser.add_argument('-c', '--calculate', default='all')
    return parser.parse_args()

def do_planning(args):
    print(args.input, args.meta, args.out)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, parse_dates=[0], index_col=0)

    with open(args.meta, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    mzp = MultiZonePlanner(df, meta, args.out)
    
    if(args.calculate == 's'):
        mzp.schedule()
    elif(args.calculate == 'sr'):
        mzp.schedule()
        mzp.roster()
    elif(args.calculate == 'srbpc'):
        mzp.schedule()
        mzp.roster()
        mzp.roster_breaks()
        mzp.roster_postprocess()
        mzp.combine_results()
    elif(args.calculate == 'pp'):
        mzp.roster_postprocess()
    else:
        mzp.solve()

def main():
    args = parse_args()
    do_planning(args)

if __name__ == '__main__':
    main()
