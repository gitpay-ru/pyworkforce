import codecs
import datetime
from pathlib import Path
import json

import pandas as pd
import numpy as np

from pyworkforce.breaks.breaks_intervals_scheduling_sat import BreaksIntervalsScheduling, AdjustmentMode
from pyworkforce.utils.solver_profile import SolverProfile
from pyworkforce.staffing.stats.calculate_stats import calculate_stats
from pyworkforce.utils.breaks_spec import build_break_spec, build_intervals_map
from pyworkforce.utils.shift_spec import get_start_from_shift_short_name, get_start_from_shift_short_name_mo, \
    required_positions, get_shift_short_name, get_shift_coverage, unwrap_shift, \
    all_zeros_shift, get_duration_from_shift_short_name, ShiftSchema
from pyworkforce.plotters.scheduling import plot_xy_per_interval
from datetime import datetime as dt
from datetime import timedelta, timezone, time, date
from pyworkforce.scheduling import MinAbsDifference
from pyworkforce.rostering.binary_programming import MinHoursRoster
from strenum import StrEnum

class Statuses(StrEnum):
    NOT_STARTED = 'NOT_STARTED',
    UNKNOWN = 'UNKNOWN',
    MODEL_INVALID = 'MODEL_INVALID',
    FEASIBLE = 'FEASIBLE',
    INFEASIBLE = 'INFEASIBLE',
    OPTIMAL = 'OPTIMAL'

    def is_ok(self):
        return self in [Statuses.OPTIMAL, Statuses.FEASIBLE]


def roll_rows(df: pd.DataFrame, count) -> pd.DataFrame:
    # roll every column,
    # this is like shift() but in a cyclic way
    for column in df:
        df[column] = np.roll(df[column], count)

    return df


def hh_mm(time_string):
    hh = int(time_string.split(":")[0])
    mm = int(time_string.split(":")[1])

    return (hh, mm)


def hh_mm_time(time_string) -> time:
    (hh, mm) = hh_mm(time_string)
    return time(hour=hh, minute=mm)


def hh_mm_timedelta(time_string) -> timedelta:
    (hh, mm) = hh_mm(time_string)
    return  timedelta(hours=hh, minutes=mm)


def get_1Day_df(time_start: time, time_end: time) -> pd.DataFrame:
    intervals = int(24*60/15)
    t = [time(hour=int(i * 15 / 60), minute=i * 15 % 60) for i in range(intervals)]

    if time_end > time_start:
        presence = [1 if (t[i] >= time_start and t[i] < time_end) else 0 for i in range(intervals)]
    else:
        presence = [1 if (t[i] >= time_start or t[i] < time_end) else 0 for i in range(intervals)]

    data = {
        "tc": t,
        "works": presence
    }

    df = pd.DataFrame(data, columns=['tc', 'works'])
    df.set_index('tc', inplace=True)
    return df


class MultiZonePlanner():
    HMin = 60
    DayH = 24

    def __init__(self,
                 df: pd.DataFrame,
                 meta: any,
                 solver_profile: any,
                 output_dir: str,
             ):

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        self.df = df

        date_diff = self.df.index[1] - self.df.index[0]
        self.step_min = int(date_diff.total_seconds() / self.HMin)
        self.days = int(self.df.index[-1].strftime("%d"))

        self.meta = meta
        # self.timezones = list(map(lambda t: int(t['utc']), self.meta['employees']))

        # shift_name -> Shift
        self.shift_data = {}  # will be filled by build_shifts()
        # shift_with_name:
        #   (id, shift_name, utc, employees_count, , employee_ratio)
        self.shift_with_names = self.build_shifts()

        # id -> (id, start_interval, endstart_interval, duration_interval)
        self.activities_by_id = self.build_activities_specs()

        # id -> (time_start_start, time_start_end, duration, time_end)
        self.shift_meta_by_id = self.build_shift_meta()

        self.shift_activities = self.build_shift_with_activities()

        self.status = Statuses.NOT_STARTED

        if solver_profile != None:
            self.solver_profile = SolverProfile.from_json(solver_profile)
        else:
            self.solver_profile = SolverProfile.default()

    def build_shift_meta(self):
        shifts = {}

        for s in self.meta['shifts']:
            hh_duration, _ = hh_mm(s['duration'])
            duration = hh_mm_timedelta(s['duration'])
            start_start = hh_mm_time(s['scheduleTimeStart'])
            start_end = hh_mm_time(s['scheduleTimeEndStart'])
            end = (dt.combine(date.today(), start_end) + duration).time()

            shifts[s['id']] = (
                start_start, start_end, duration, end
            )

        return shifts

    def build_capacity_df(self) -> pd.DataFrame:
        campaign_utc = self.meta['campainUtc']

        dfs = []
        for party in self.shift_with_names:
            (shift_id, shift_name, employee_utc, employee_count, schema) = party
            (shift_start, *_, shift_end) = self.shift_meta_by_id[shift_id]

            delta_utc = campaign_utc - employee_utc
            df = get_1Day_df(shift_start, shift_end)  # "tc": time, "works": int
            df[shift_name] = df['works'] * employee_count

            # make utc shift from employee local daytime to campaign datetime
            df = roll_rows(df, delta_utc * 4)
            dfs.append(df[[shift_name]])

        df_shifts = pd.concat(dfs, axis=1)
        df_shifts['total'] = df_shifts.sum(axis=1)

        return df_shifts

    def build_shifts(self):
        edf = pd.DataFrame(self.meta['employees'])
        edf['schema'] = edf.apply(lambda t: t['schemas'][0], axis=1)
        edf['shiftId'] = edf.apply(lambda t: self.get_shift_by_schema(t['schema']), axis=1)
        edf_g = edf.groupby(['utc', 'shiftId', 'schema'])['id'].agg(['count'])
        shift_with_names = []

        # [('c8e4261e-3de3-4343-abda-dc65e4042494', '+6', 150, 'x_9_6_13_15', 0.410958904109589), ('c8e4261e-3de3-4343-abda-dc65e4042495', '+3', 33, 'x_9_6_13_15', 0.09041095890410959), ('c8e4261e-3de3-4343-abda-dc65e4042490', '+3', 32, 'x_12_6_13_15', 0.08767123287671233), ('22e4261e-3de3-4343-abda-dc65e4042496', '-3', 150, 'x_9_6_13_15', 0.410958904109589)]
        for index, row in edf_g.iterrows():
            utc = index[0]
            shift_orig_id = index[1]
            schema_name = index[2]
            shift_name = self.get_shift_name_by_id(shift_orig_id, utc)
            employee_count = int(row['count'])  # by default its int64 -> non serializible

            shift_with_names.append(
                (shift_orig_id, shift_name, utc, employee_count, schema_name,)
            )

            meta_shift = next(t for t in self.meta['shifts'] if t['id'] == shift_orig_id)
            meta_schema = next(t for t in self.meta['schemas'] if t['id'] == schema_name)

            self.shift_data[shift_name] = ShiftSchema(
                shift_name = shift_name,
                shift_id=shift_orig_id,
                schema_id=schema_name,
                utc=utc,
                min_start_time=meta_shift['scheduleTimeStart'],
                max_start_time=meta_shift['scheduleTimeEndStart'],
                duration_time=meta_shift['duration'],
                holidays_min=meta_schema['holidays']['minDaysInRow'],
                holidays_max=meta_schema['holidays']['maxDaysInRow'],
                work_min=meta_schema['shifts'][0]['minDaysInRow'],
                work_max=meta_schema['shifts'][0]['maxDaysInRow'],
                employee_count=employee_count
            )

        return shift_with_names

    def build_activities_specs(self):
        activities_specs = build_break_spec(self.meta)
        #   "9 часов день обед" -> ("9 часов день обед", "02:00", "07:00", "00:15")
        activities_map = {**{b: (b, *_) for (b, *_) in activities_specs}}

        return activities_map

    def build_shift_with_activities(self):
        (m, _) = build_intervals_map()
        shifts = {}

        for s in self.meta["shifts"]:
            s_id = s["id"]
            activities = s["activities"]
            min_between = m[s["minIntervalBetweenActivities"]]
            max_between = m[s["maxIntervalBetweenActivities"]]

            # (activities_id[], min_interval_between, max_interval_between)
            shifts[s_id] = (activities, min_between, max_between)

        return shifts

    def solve(self):
        self.status = Statuses.NOT_STARTED

        # 0. Calculate required positions to shedule
        self.calc_required_positions()

        # 1. Schedule shift
        self.schedule()
        if not self.status.is_ok():
            return self.status

        # 2. Assign resources per shifts
        self.roster()
        if not self.status.is_ok():
            return self.status

        # 3. Roster breaks
        self.roster_breaks()

        # 4. Process results before return
        self.roster_postprocess()

        # 5. Combine all in one json
        self.combine_results()

        # 6. Recalculate statistics
        self.recalculate_stats()

        # return the latest status of the rostering model
        # should be either OPTIMAL or FEASIBLE
        return self.status

    def build_stats_df(self):

        campaign_utc = int(self.meta['campainUtc'])
        campaign_tz = timezone(timedelta(hours=campaign_utc))

        # just a helper function to use
        def replace_nan(df, col, what):
            nans = df[col].isnull()
            df.loc[nans, col] = [what for isnan in nans.values if isnan]
            return df

        def to_df_stats(df: pd.DataFrame):
            df.reset_index(inplace=True)
            df['tc'] = df_total['tc'].dt.tz_localize(campaign_tz)
            df_total['tc'] = df_total['tc'].dt.strftime('%Y-%m-%d %H:%M:%S%z')

            interested_columns = ['tc', 'call_volume', 'aht', 'service_level', 'art', 'positions', 'resources_shifts']

            df = df[interested_columns]
            df = df.rename(columns={'resources_shifts': 'scheduled_positions'})

            return df

        # prepate data for further sums
        df_total = pd.read_csv(f'{self.output_dir}/required_positions.csv', encoding='utf-8')
        df_total['tc'] = pd.to_datetime(df_total['tc'])
        df_total['tc'] = df_total['tc'].dt.tz_localize(None)
        df_total['resources_shifts'] = np.zeros(shape=len(df_total))
        df_total['frac'] = np.zeros(shape=len(df_total))
        df_total['positions_quantile'] = np.zeros(shape=len(df_total))
        df_total.set_index('tc', inplace=True)
        df_total.sort_index(inplace=True)

        # This is virtual empty shift, to be used as a filler for rest days
        empty_shift = np.array(all_zeros_shift()) * 1
        empty_schedule = pd.DataFrame(index=[i for i in range(self.days)])
        periods_in_hour = 4

        for party in self.shift_with_names:
            (shift_id, shift_name, utc, *_) = party

            print(f'Shift: {shift_name} ({shift_id})')

            utc_shift = int(utc) - campaign_utc

            # Load breaks and converto to df
            # breaks are in the Shift (=employee) utc
            with open(f'{self.output_dir}/breaks_output_{shift_name}.json', 'r', encoding='utf-8') as f:
                breaks = json.load(f)
            list_breaks = self.get_breaks_intervals_per_slot(breaks['resource_break_intervals'])
            df_breaks = pd.DataFrame(list_breaks, columns=["resource", "day", "breaks"])
            df_breaks.set_index(["resource", "day"], inplace=True)

            # Load rostering data
            # Rostering is in the Shift (=employee) utc
            with open(f'{self.output_dir}/rostering_output_{shift_name}.json', 'r', encoding='utf-8') as f:
                rostering = json.load(f)
            df = pd.DataFrame(rostering['resource_shifts'])

            # Rostering - breaks = schedule
            df['shifted_resources_per_slot'] = df.apply(
                lambda t: np.array(unwrap_shift(t['shift'])) * 1 - df_breaks.loc[str(t['resource']), t['day']][0],
                axis=1
            )

            df1 = df[['day', 'shifted_resources_per_slot']]\
                .groupby('day', as_index=True)['shifted_resources_per_slot']\
                .apply(lambda x: np.sum(np.vstack(x), axis=0))\
                .to_frame()

            # on missed indexes (=days), NaN will be placed, because there are no any rest days in df1
            df1 = pd.concat([df1, empty_schedule], axis=1)
            df1 = replace_nan(df1, 'shifted_resources_per_slot', empty_shift)
            # new items are at the end with propper index - just sort them to be moved to correct position
            df1 = df1.sort_index(ascending=True)

            np.set_printoptions(linewidth=np.inf, formatter=dict(float=lambda x: "%3.0i" % x))
            arr = df1['shifted_resources_per_slot'].values
            arr = np.concatenate(arr)

            df3 = pd.read_csv(f'{self.output_dir}/required_positions_{shift_name}.csv', encoding='utf-8')
            # todo: Store intermediate files either with correct (=shift) TZ or without any TZ at all
            df3['tc'] = pd.to_datetime(df3['tc'])
            df3['tc'] = df3['tc'].dt.tz_localize(None)
            df3.set_index('tc', inplace=True)
            df3.sort_index()  # just to be on a safe side
            # df3.reset_index(inplace=True)

            df3['resources_shifts'] = arr.tolist()

            # just copy some rows for further vaerification
            df3 = df3.shift(periods = -1 * utc_shift*periods_in_hour, fill_value = 0)

            if df_total is None:
                # this is to just copy required positions
                # todo: invent pre-schedule step and persist self.df results to .csv + load it here into df_total
                df_total = df3
            else:
                df_total['resources_shifts'] += df3['resources_shifts']
                df_total['frac'] += df3['frac']
                df_total['positions_quantile'] += df3['positions_quantile']

        # final formating
        df_total.reset_index(inplace=True)
        df_total['tc'] = df_total['tc'].dt.tz_localize(campaign_tz)
        df_total['tc'] = df_total['tc'].dt.strftime('%Y-%m-%d %H:%M:%S%z')

        df_total = df_total[['tc', 'call_volume', 'aht', 'service_level', 'art', 'positions', 'resources_shifts']]
        df_total['positions'] = df_total['positions'].astype(int)
        df_total['resources_shifts'] = df_total['resources_shifts'].astype(int)

        df_total = df_total.rename(columns={'resources_shifts': 'scheduled_positions'})

        return df_total

    def dump_stat_and_plot(self, shift_name, solution, df):
        resources_shifts = solution['resources_shifts']
        df3 = pd.DataFrame(resources_shifts)
        df3['shifted_resources_per_slot'] = df3.apply(lambda t: np.array(unwrap_shift(t['shift'])) * t['resources'], axis=1)
        df4 = df3[['day', 'shifted_resources_per_slot']].groupby('day', as_index=False)['shifted_resources_per_slot'].apply(lambda x: np.sum(np.vstack(x), axis = 0)).to_frame()
        np.set_printoptions(linewidth=np.inf, formatter=dict(float=lambda x: "%3.0i" % x))
        df4.to_csv(f'{self.output_dir}/shifted_resources_per_slot_{shift_name}.csv')
        arr = df4['shifted_resources_per_slot'].values
        arr = np.concatenate(arr)
        df['resources_shifts'] = arr.tolist()
        df.to_csv(f'{self.output_dir}/scheduling_output_stage2_{shift_name}.csv')

        plot_xy_per_interval(f'{self.output_dir}/scheduling_{shift_name}.png', df, x='index', y=["positions", "resources_shifts"])

    def plot_scheduling(self, schedule_results):
        periods_in_hour = 4
        campaign_utc = self.meta['campainUtc']

        df_sum = None

        for (shift_name, shift_utc, solution, df) in schedule_results:
            utc_delta = shift_utc - campaign_utc
            resources_shifts = solution['resources_shifts']

            df3 = pd.DataFrame(resources_shifts)
            df3['shifted_resources_per_slot'] = df3.apply(lambda t: np.array(unwrap_shift(t['shift'])) * t['resources'], axis=1)

            df4 = df3[['day', 'shifted_resources_per_slot']]\
                .groupby('day', as_index=False)['shifted_resources_per_slot']\
                .apply(lambda x: np.sum(np.vstack(x), axis = 0))\
                .to_frame()

            arr = df4['shifted_resources_per_slot'].values
            arr = np.concatenate(arr)

            df['resources_shifts'] = arr.tolist()

            df.reset_index(inplace=True)
            # make it tz-agnopdtic
            df['tc'] = pd.to_datetime(df['tc'])
            df['tc'] = df['tc'].dt.tz_localize(None)
            df.set_index('tc', inplace=True)
            df.sort_index(inplace=True)
            # anf shift rows according to tz delta, this won't shift the index itself
            df = df.shift(periods=-1 * utc_delta * periods_in_hour, fill_value=0)

            if df_sum is None:
                df_sum = df.copy()
            else:
                df_sum['positions_quantile'] += df['positions_quantile']
                df_sum['resources_shifts'] += df['resources_shifts']

        plot_xy_per_interval(f'{self.output_dir}/scheduling.png', df_sum, x='index', y=["positions", "resources_shifts"])

    def dump_scheduling_output_rostering_input(self, shift_suffix, shift_id, days, num_resources, solution, shifts_spec):
        with open(f'{self.output_dir}/scheduling_output_{shift_suffix}.json', 'w') as f:
                f.write(json.dumps(solution, indent=2))

        resources_shifts = solution['resources_shifts']
        df1 = pd.DataFrame(resources_shifts)
        df2 = df1.pivot(index='shift', columns='day', values='resources').rename_axis(None, axis=0)

        df2['combined']= df2.values.tolist()

        rostering = {
            'num_days': days,
            'num_resources': num_resources,
            'shift_id': shift_id,
            'shifts': list(shifts_spec.keys()),
            # 'min_working_hours': 176,  # Dec 2022 #todo:
            # 'max_resting': 9,  # Dec 2022
            # 'non_sequential_shifts': [],
            'required_resources': df2['combined'].to_dict(),
            # 'banned_shifts': [],
            # 'resources_preferences': [],
            # 'resources_prioritization': []
        }

        with open(f'{self.output_dir}/scheduling_output_rostering_input_{shift_suffix}.json', 'w') as outfile:
            outfile.write(json.dumps(rostering, indent=2))

    def get_shift_by_schema(self, schema_id):
        schema = next(t for t in self.meta['schemas'] if t['id'] == schema_id)
        shift_id = schema['shifts'][0]['shiftId']
        return shift_id

    def get_shift_size(self, shift_id):
        shift = next(t for t in self.meta['shifts'] if t['id'] == shift_id)
        return dt.strptime(shift['duration'], "%H:%M").hour

    def get_activities_by_schema(self, schema_id):
        shift_id = self.get_shift_by_schema(schema_id)
        shift = next(t for t in self.meta['shifts'] if t['id'] == shift_id)

        cx = 0
        # lambda t: format(dt.strptime(t['activityTimeStart'], "%H:%M") + timedelta(hours=delta), '%H:%M'),
        for activity_id in shift['activities']:
            activity = next(t for t in self.meta['activities'] if t['id'] == activity_id)
            if activity != None:
                cx += dt.strptime(activity['duration'], "%H:%M").minute / 60.0
        return cx

    def get_unpaid_activities_by_shift(self, shift_id):
        shift = next(t for t in self.meta['shifts'] if t['id'] == shift_id)
        cx = 0
        for activity_id in shift['activities']:
            activity = next(t for t in self.meta['activities'] if t['id'] == activity_id)
            if activity != None:
                if not activity['isPaid']:
                    h = dt.strptime(activity['duration'], "%H:%M").hour
                    m = dt.strptime(activity['duration'], "%H:%M").minute
                    cx += h + m / 60.0
        return cx

    def get_shift_name_by_id(self, id, utc):
        shift = next(t for t in self.meta['shifts'] if t['id'] == id)
        shift_name = get_shift_short_name(shift, utc)
        return shift_name

    def get_breaks_intervals_per_slot(self, resource_break_intervals: dict):
        # "resource" ->  [(day_num, break_id, start, end)]
        _days = self.days
        _interval_per_hour = 4
        empty_month = np.zeros(_days * 24 * _interval_per_hour).astype(int)
        _eom = len(empty_month)

        # output: [ (resource(string), day, [010101010101]) ]
        result = []

        for resource_id, bi in resource_break_intervals.items():
            resource_month = empty_month.copy()
            for (day_num, break_id, start, end) in bi:
                # breaks are calculated with overnights also
                # => for the last day of month it could plan for a day after that.
                if start > _eom:
                    continue
                resource_month[start:end] = [1 for _ in range(start, min(end, _eom))]

            for d in range(_days):
                day_start = d * 24 * _interval_per_hour
                day_end_exclusive = (d + 1) * 24 * _interval_per_hour
                result.append(
                    (str(resource_id), int(d), resource_month[day_start:day_end_exclusive])
                )

        return result

    def get_breaks_per_day(self, resource_break_intervals: dict):
        # "resource" ->  [(day_num, break_id, start, end)]
        _days = self.days
        _interval_per_hour = 4
        _full_day = 24*_interval_per_hour
        _eom = _days * _full_day

        (_, t) = build_intervals_map()

        # output: [ (resource, day, break_id, start_time, end_time ]
        result = []

        for resource_id, bi in resource_break_intervals.items():
            for (day_num, break_id, start, end) in bi:
                # breaks are calculated with overnights also
                # => for the last day of month it could plan for a day after that.
                if start > _eom:
                    continue

                # day_n = int(start/_full_day)
                day_n = day_num

                start_from_day = start - day_n*_full_day
                # for overnight intervals -> return next day's time
                if (start_from_day >= _full_day):
                    start_from_day -= _full_day

                end_from_day_inclusive = end - day_n*_full_day
                #end_from_day_inclusive = (end - 1) - day_n * _full_day
                if (end_from_day_inclusive >= _full_day):
                    end_from_day_inclusive -= _full_day

                start_time = t[start_from_day]
                end_time = t[end_from_day_inclusive]

                result.append(
                    (str(resource_id), day_n, break_id, start_time, end_time)
                )

        return result

    @property
    def ts(self):
        HMin = 60
        DayH = 24

        date_diff = self.df.index[1] - self.df.index[0]
        step_min = int(date_diff.total_seconds() / HMin)
        ts = int(HMin / step_min)

        return ts

    def calc_required_positions(self):
        print("Start calculating required positions")

        self.df['positions'] = self.df.apply(lambda row: required_positions(
            call_volume=row['call_volume'],
            aht=row['aht'],
            interval=15 * 60,  # interval should be passed within the same dimension as aht & art
            art=row['art'],
            service_level=row['service_level']
        ), axis=1)

        campaign_utc = int(self.meta['campainUtc'])
        campaign_tz = timezone(timedelta(hours=campaign_utc))

        self.df.index = self.df.index.tz_localize(tz=campaign_tz)
        self.df.to_csv(f'{self.output_dir}/required_positions.csv', encoding='utf-8')

        # 'df_capacity' is in a campaign tz
        df_capacity = self.build_capacity_df()

        for party in self.shift_with_names:
            (shift_id, shift_name, shift_utc, employee_count, schema) = party
            print(shift_name)

            utc_delta = int(shift_utc) - campaign_utc
            shift_tz = timezone(timedelta(hours=int(shift_utc)))

            # 'df' is in a campaign tz
            df = self.df.copy()

            df_my_capacity = df_capacity[[shift_name, 'total']].copy()
            df_my_capacity.rename(columns={shift_name: 'capacity'}, inplace=True)
            # df_my_capacity['capacity'] = df_my_capacity['capacity'].astype(int)
            df_my_capacity['frac'] = df_my_capacity['capacity'] / df_my_capacity['total']
            df_my_capacity['frac'].fillna(0.0, inplace=True)  # if after division == NaN -> 0.0

            df = df.reset_index()
            df['time'] = df['tc'].dt.time
            df.set_index('time', inplace=True)

            # since capacity is by time (not by datetime!!!) - then need to merge by time portion only
            df = pd.merge(df, df_my_capacity[['frac', 'capacity']], how='inner', left_index=True, right_index=True)
            df = df.reset_index(drop=True)  # don't need 'time' index anymore, was used got joins only
            df.set_index('tc', inplace=True)
            df.sort_index(inplace=True)

            df['positions_quantile'] = np.ceil(df['positions'] * df['frac']).astype(int)

            # tz_convert will make shift starting from 01:00 (e.g. when switching from +3 to +4)
            # but we need to keep the schedule starting from 00:00
            # df.index = df.index.tz_convert(tz=shift_tz)

            df.index = df.index.tz_localize(None)  # 1. switch off timezones
            # 2. roll (=erase) data to match statistics to the target tz
            if utc_delta != 0:
                # original idea - use np.roll to avoid replacing with zeros
                # roll records down: 00h in +3 tz, is 01h in +4 tz
                df = roll_rows(df, self.ts * utc_delta)
            df.index = df.index.tz_localize(shift_tz)  # 3. switch on new tz

            df.to_csv(f'{self.output_dir}/required_positions_{shift_name}.csv', encoding='utf-8')

        print("Done calculating required positions")

    def schedule(self):
        print("Start scheduling")

        DayH = 24
        min_date = min(self.df.index)
        max_date = max(self.df.index)
        days = (max_date - min_date).days + 1

        schedule_results = []

        for party in self.shift_with_names:
            (shift_id, shift_name, utc, employee_count, schema) = party

            shift_names = [shift_name]
            shifts_coverage = get_shift_coverage(shift_names)

            # required positions .csv is in a shift timezone already
            df = pd.read_csv(f'{self.output_dir}/required_positions_{shift_name}.csv')
            df.set_index('tc', inplace=True)

            required_resources = []
            capacity = []

            for i in range(self.days):
                df_short = df[i * self.DayH * self.ts : (i + 1) * self.DayH * self.ts]
                required_resources.append(df_short['positions_quantile'].tolist())
                capacity.append(df_short['capacity'].tolist())

            scheduler = MinAbsDifference(num_days = self.days,  # S
                                periods = self.DayH * self.ts,  # P
                                shifts_coverage = shifts_coverage,
                                required_resources = required_resources,
                                # max_period_concurrency=capacity,  # gamma
                                max_period_concurrency = int(df['positions_quantile'].max()),  # gamma
                                # max_shift_concurrency=int(df['positions_quantile'].mean()),  # beta
                                max_shift_concurrency=employee_count,  # beta
                                solver_params=self.solver_profile.scheduler_params
                                )

            solution = scheduler.solve()

            # if solution not feasible -> stop it and return result
            self.status = Statuses(solution['status'])
            if not self.status.is_ok():
                return

            self.dump_scheduling_output_rostering_input(
                shift_name,
                shift_id,
                self.days,
                employee_count,
                solution,
                shifts_coverage
            )

            self.dump_stat_and_plot(
                shift_name,
                solution,
                df.copy()
            )

            schedule_results.append(
                (shift_name, utc, solution,  df.copy())
            )

        self.plot_scheduling(schedule_results)

        return "Done scheduling"

    def roster(self, skip_existing: bool = False):

        CHUNK_SIZE = 1

        def chunker(seq, size):
            if size > 0:
                return (seq[pos:pos + size] for pos in range(0, len(seq), size))
            else:
                return seq

        def aggregate_solutions(agg, value):
            if agg is None:
                return value

            agg['cost'] += value['cost']
            agg['shifted_hours'] += value['shifted_hours']
            agg['total_resources'] += value['total_resources']
            agg['total_shifts'] += value['total_shifts']
            agg['resting_days'] += value['resting_days']
            agg['resource_shifts'] += value['resource_shifts']
            agg['resting_resource'] += value['resting_resource']

            return agg

        def reduce_positions(main: dict, reducer: dict):
            # shift_name -> [list of scheduled positions]
            result = {}
            for s in main:
                result[s] = [0 if i==0 else (i-j) for i,j in zip(main[s], reducer[s])] # to avoid required negatives

            return result

        def resource_shifts_to_shifts_positions(days:int, shift_names:list, resource_shifts:dict):
            # resource_shifts:
            # [{'id': 0, 'resource': 'c1fcffd5-1eb7-c38b-7209-a6106d66cc84', 'day': 1, 'shift': 'x_9_12_45'},
            #  {'id': 0, 'resource': 'c1fcffd5-1eb7-c38b-7209-a6106d66cc84', 'day': 2, 'shift': 'x_9_12_45'},
            #  {'id': 0, 'resource': 'c1fcffd5-1eb7-c38b-7209-a6106d66cc84', 'day': 3, 'shift': 'x_9_12_45'},
            #   ...
            #  ]

            shift_positions = {}

            for s in shift_names:
                shift_positions[s] = [0 for _ in range(days)]

            for rs in resource_shifts:
                shift = rs['shift']
                day = rs['day']
                shift_positions[shift][day] += 1

            return shift_positions

        print(f"Start rostering, skip existing calcs = {skip_existing}")
        for party in self.shift_with_names:
            (shift_id, shift_name, utc, *_) = party
            print(f'Shift: {shift_name} {shift_id}')

            if Path(f'{self.output_dir}/rostering_output_{shift_name}.json').exists() and skip_existing:
                print(f'Shift: {self.output_dir}/rostering_output_{shift_name}.json found, skipping it')
                continue

            with open(f'{self.output_dir}/scheduling_output_rostering_input_{shift_name}.json', 'r') as f:
                shifts_info = json.load(f)

            shift_names = shifts_info["shifts"]
            unpaid_hours = self.get_unpaid_activities_by_shift(shifts_info['shift_id'])
            shifts_hours = [int(i.split('_')[1]) - unpaid_hours for i in shifts_info["shifts"]]

            edf = pd.DataFrame(self.meta['employees'])
            edf['shiftId'] = edf.apply(lambda t: self.get_shift_by_schema(t['schemas'][0]), axis=1)

            edf_filtered = edf[(edf['utc'] == utc) & (edf['shiftId'] == shift_id)]
            print(edf_filtered)

            resources = list(edf_filtered['id'])
            resources_min_w_hours = list(edf_filtered['minWorkingHours'])
            resources_max_w_hours = list(edf_filtered['maxWorkingHours'])
            print(f'Rostering num: {shifts_info["num_resources"]} {len(resources)}')

            shift_data: ShiftSchema = self.shift_data[shift_name]

            # constraint:
            #   (hard_min, soft_min, penalty, soft_max, hard_max, penalty)
            work_constraints = [
                # e.g.: no low bound, optimum - from 5 to 5 without penalty, more than 5 are forbidden
                # (0, 5, 0, 5, 5, 0)

                # work at least 'work_min', but no more than 'work_max',
                # 'work_max' is both lower and upper soft intertval -> deltas are penalized by 1
                (shift_data.work_min, shift_data.work_min, 0, shift_data.work_max, shift_data.work_max, 0)
            ]

            rest_constraints = [
                # 1 to 3 non penalized holidays
                (shift_data.holidays_min, shift_data.holidays_min, 0, shift_data.holidays_max, shift_data.holidays_max, 0)
            ]

            num_days = shifts_info["num_days"]
            shifts = shifts_info["shifts"]
            required_resources = shifts_info["required_resources"]

            agg_solution = None

            print(f"Solving by chunks, CHUNK_SIZE = {CHUNK_SIZE}")

            for i, chunk in enumerate(zip(chunker(resources, CHUNK_SIZE),
                                          chunker(resources_min_w_hours, CHUNK_SIZE),
                                          chunker(resources_max_w_hours, CHUNK_SIZE))):

                print(f"Chunk #{i}")

                (resources_chunk, min_w_hours_chunk, max_w_hours_chunk) = chunk

                solver = MinHoursRoster(num_days=num_days,
                                        resources=resources_chunk,
                                        resources_min_w_hours = min_w_hours_chunk,
                                        resources_max_w_hours = max_w_hours_chunk,
                                        shifts=shifts,
                                        shifts_hours=shifts_hours,
                                        required_resources=required_resources,
                                        shift_constraints=work_constraints,
                                        rest_constraints=rest_constraints,
                                        solver_params=self.solver_profile.roster_params
                                        )

                solution = solver.solve()

                # if solution not feasible -> stop it and return result
                self.status = Statuses(solution['status'])
                if not self.status.is_ok():
                    raise Exception(f'Rostering for {shift_name} failed with status {self.status}')

                shift_positions = resource_shifts_to_shifts_positions(num_days, shifts, solution['resource_shifts'])

                agg_solution = aggregate_solutions(agg_solution, solution)
                required_resources = reduce_positions(required_resources, shift_positions)

            with open(f'{self.output_dir}/rostering_output_{shift_name}.json', 'w') as f:
                f.write(json.dumps(agg_solution, indent = 2))

        print("Done rostering")
        return


    def roster_breaks(self):
        print("Start breaks rostering")
        (m, _) = build_intervals_map()

        def daily_start_index(day):
            return day*24*4

        def get_working_intervals(edf: pd.DataFrame):
            edf["day_index"] = edf.apply(lambda row: daily_start_index(row["day"]), axis=1)
            edf["start_interval"] = \
                edf["day_index"] + edf.apply(lambda row: m[get_start_from_shift_short_name_mo(row["shift"])], axis=1)
            edf["duration_interval"] = edf.apply(lambda row: get_duration_from_shift_short_name(row["shift"]), axis=1) * 4  # durations are in hours
            edf["end_interval"] = edf["start_interval"] + edf["duration_interval"]

            return edf[["day", "start_interval", "end_interval"]].to_records(index=False).tolist()

        # 0. iterate over known shifts, breaks are same for employees within given shift
        for party in self.shift_with_names:
            (shift_id, shift_name, utc, *_) = party

            print(f'Shift: {shift_name} ({shift_id})')

            # 1. Summarize breaks details
            (breaks_ids, min_delay, max_delay) = self.shift_activities[shift_id]
            breaks_specs = [self.activities_by_id[b] for b in breaks_ids]

            # 2. Rostering gives Employee' schedules
            with open(f'{self.output_dir}/rostering_output_{shift_name}.json', 'r') as f:
                rostering = json.load(f)

            df = pd.DataFrame(rostering['resource_shifts'])
            employee_schedule = {}
            for (index, df_e) in df.groupby(["resource"]):
                employee_schedule[index] = get_working_intervals(df_e)

            # 3. Run model
            model = BreaksIntervalsScheduling(
                employee_calendar=employee_schedule,
                breaks=breaks_specs,
                break_min_delay=min_delay,
                break_max_delay=max_delay,
                make_adjustments=AdjustmentMode.ByExpectedAverage,
                solver_params=self.solver_profile.breaks_params
            )

            solution = model.solve()

            self.status = Statuses(solution['status'])
            if not self.status.is_ok():
                raise Exception(f'Breakes for {shift_name} failed with status {self.status}')

            with codecs.open(f'{self.output_dir}/breaks_output_{shift_name}.json', 'w', encoding='utf-8') as f:
                json.dump(solution, f, indent=2, ensure_ascii=False)


        print("Done breaks rostering")
        return "Done"


    def roster_postprocess(self):
        print("Start rostering postprocessing")

        # just a helper function to use
        def replace_nan(df, col, what):
            nans = df[col].isnull()
            df.loc[nans, col] = [what for isnan in nans.values if isnan]
            return df

        df_total: pd.DataFrame = None

        for party in self.shift_with_names:
            (shift_id, shift_name, utc, *_) = party

            print(f'Shift: {shift_name} ({shift_id})')

            # Load breaks and converto to df
            with open(f'{self.output_dir}/breaks_output_{shift_name}.json', 'r', encoding='utf-8') as f:
                breaks = json.load(f)
            list_breaks = self.get_breaks_intervals_per_slot(breaks['resource_break_intervals'])
            df_breaks = pd.DataFrame(list_breaks, columns=["resource", "day", "breaks"])
            df_breaks.set_index(["resource", "day"], inplace=True)

            # Load rostering data
            with open(f'{self.output_dir}/rostering_output_{shift_name}.json', 'r') as f:
                rostering = json.load(f)
            df = pd.DataFrame(rostering['resource_shifts'])

            # This is virtual empty shift, to be used as a filler for rest days
            empty_shift = np.array(all_zeros_shift()) * 1
            empty_schedule = pd.DataFrame(index = [i for i in range(self.days)])

            # Rostering - breaks = schedule
            df['shifted_resources_per_slot'] = df.apply(
                lambda t: np.array(unwrap_shift(t['shift'])) * 1 - df_breaks.loc[str(t['resource']), t['day']][0], axis=1
            )

            df1 = df[['day', 'shifted_resources_per_slot']].groupby('day', as_index=True)[
                'shifted_resources_per_slot'].apply(lambda x: np.sum(np.vstack(x), axis=0)).to_frame()

            # on missed indexes (=days), NaN will be placed, because there are no any rest days in df1
            df1 = pd.concat([df1, empty_schedule], axis=1)
            df1 = replace_nan(df1, 'shifted_resources_per_slot', empty_shift)
            # new items are at the end with propper index - just sort them to be moved to correct position
            df1 = df1.sort_index(ascending=True)

            np.set_printoptions(linewidth=np.inf, formatter=dict(float=lambda x: "%3.0i" % x))
            arr = df1['shifted_resources_per_slot'].values
            arr = np.concatenate(arr)

            df3 = pd.read_csv(f'{self.output_dir}/required_positions_{shift_name}.csv')
            df3['resources_shifts'] = arr.tolist()
            plot_xy_per_interval(f'{self.output_dir}/rostering_{shift_name}.png', df3, x='index', y=["positions_quantile", "resources_shifts"])

            if df_total is None:
                df_total = df3
            else:
                df_total['resources_shifts'] += df3['resources_shifts']

        plot_xy_per_interval(f'{self.output_dir}/rostering.png', df_total, x='index', y=["positions", "resources_shifts"])

        print("Done rostering postprocessing")
        return "Done"

    def combine_results(self):
        print(f'Start combining results')

        def time_str_to_datetime(time_str, day, month, year, tz, format):
            t = dt.strptime(time_str, format)
            return datetime.datetime(year=year, month=month, day=day, hour=t.hour, minute=t.minute, second=t.second, tzinfo=tz)

        def shift_name_to_datetime(shift, day, month, year, tz):
            time_str = get_start_from_shift_short_name(shift)
            return time_str_to_datetime(time_str, day, month, year, tz, format="%H:%M:%S")


        campain_utc = self.meta['campainUtc']
        campaign_tz = timezone(timedelta(hours=campain_utc))

        min_date = min(self.df.index)  # 2023-03-01 00:00:00 (TimeStamp)
        m = min_date.month
        y = min_date.year

        out = {
            "campainUtc": campain_utc,
            "campainSchedule": []
        }

        campainSchedule = out['campainSchedule']
        for party in self.shift_with_names:
            (shift_name, shift_code, shift_utc, mp, schema_name) = party
            shift_tz = timezone(timedelta(hours=shift_utc))

            print(f'Shift: {shift_code} ({shift_name})')

            with open(f'{self.output_dir}/rostering_output_{shift_code}.json', 'r', encoding='utf-8') as f:
                rostering = json.load(f)

            df = pd.DataFrame(rostering['resource_shifts'])
            df['shiftTimeStartLocal'] = df.apply(lambda t: shift_name_to_datetime(t['shift'], (t['day'] + 1), m, y, shift_tz), axis=1)
            df['shiftTimeStartLocal'] = df['shiftTimeStartLocal'].dt.tz_convert(tz=campaign_tz)
            df['schemaId'] = schema_name
            df['shiftId'] = shift_name
            df['employeeId'] = df['resource']
            df['employeeUtc'] = shift_utc
            df['shiftTimeStart'] = df['shiftTimeStartLocal'].dt.time
            df['shiftDate'] = df['shiftTimeStartLocal'].dt.strftime('%d.%m.%y')

            # Load breaks and converto to df
            with open(f'{self.output_dir}/breaks_output_{shift_code}.json', 'r', encoding='utf-8') as f:
                breaks = json.load(f)
            list_breaks = self.get_breaks_per_day(breaks['resource_break_intervals'])
            df_breaks = pd.DataFrame(list_breaks,
                                     columns=["resource", "day", "activityId", "activityTimeStart", "activityTimeEnd"])

            # df_breaks['activityTimeStart'] = df_breaks.apply(
            #     lambda t: format(dt.strptime(t['activityTimeStart'], "%H:%M") - timedelta(hours=delta), '%H:%M'), axis=1)
            df_breaks['activityTimeStartLocal'] = df_breaks.apply(
                lambda t: time_str_to_datetime(t['activityTimeStart'], t['day'] + 1, m, y, shift_tz, format="%H:%M"), axis=1)
            df_breaks['activityTimeStart'] = df_breaks['activityTimeStartLocal'].dt.tz_convert(tz=campaign_tz).dt.time

            # df_breaks['activityTimeEnd'] = df_breaks.apply(
            #     lambda t: format(dt.strptime(t['activityTimeEnd'], "%H:%M") - timedelta(hours=delta), '%H:%M'), axis=1)
            df_breaks['activityTimeEndLocal'] = df_breaks.apply(
                lambda t: time_str_to_datetime(t['activityTimeEnd'], t['day'] + 1, m, y, shift_tz, format="%H:%M"),axis=1)
            df_breaks['activityTimeEnd'] = df_breaks['activityTimeEndLocal'].dt.tz_convert(tz=campaign_tz).dt.time

            df_breaks.set_index(["resource", "day"], inplace=True)
            df_breaks = df_breaks[['activityId', 'activityTimeStart', 'activityTimeEnd']]
            df['activities'] = df.apply(
                lambda t: df_breaks[df_breaks.index.isin([(str(t['employeeId']), t['day'])])], axis=1)

            res = json.loads(df[['employeeId', 'employeeUtc', 'schemaId', 'shiftId', 'shiftDate', 'shiftTimeStart',
                                 'activities']].to_json(orient="records"))
            campainSchedule.extend(res)

        with open(f'{self.output_dir}/rostering.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(out, indent=2, ensure_ascii=False))

        print(f'Done combining results')

    def recalculate_stats(self):
        print("Start calculating statistics")

        df_stats = self.build_stats_df()
        df_stats = calculate_stats(df_stats)

        # dump statistics to .json
        result = df_stats.to_json(orient="records")
        parsed = json.loads(result)

        print("Writing statistics to .json")

        with open(f'{self.output_dir}/statistics_output.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(parsed, indent=2))

        print("Done calculating statistics")

