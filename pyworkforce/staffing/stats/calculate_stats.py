import datetime

import numpy as np
import pandas
from pyworkforce import ErlangC
from pyworkforce.utils.shift_spec import get_shift_coverage


def get_datetime(t):
    return datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')


def positions_service_level(call_volume, aht, interval, art, positions):
    if call_volume ==0:
        return (0.0, 0.0, 0.0)

    erlang = ErlangC(transactions=call_volume, aht=aht, interval=interval, asa=art, shrinkage=0.0)

    zero_level_positions = erlang.required_positions(0.0)["positions"]

    if positions <= 0:
        return (0.0, 0.0, zero_level_positions)

    asa = erlang.what_asa(positions)
    if asa <=0:
        return (0.0, asa, zero_level_positions)

    service_level = erlang.service_level(positions, scale_positions=False, asa=asa) * 100

    # (service_level, asa)
    return (service_level, asa, zero_level_positions)


def calculate_stats(df: pandas.DataFrame):

    # ['tc', 'call_volume', 'service_level', 'art', 'positions', 'scheduled_positions']

    for i in range(len(df)):
        (sl, asa, zero_level_positions) = positions_service_level(
            call_volume=df.loc[i, 'call_volume'],
            aht = df.loc[i, 'aht'],
            interval=15*60,
            art=df.loc[i, 'art'],
            positions=df.loc[i, 'scheduled_positions'])

        df.loc[i, 'scheduled_service_level'] = round(sl, 2)
        df.loc[i, 'scheduled_asa'] = round(asa, 2)
        df.loc[i, 'zero_level_positions'] = round(zero_level_positions, 2)

    return df
