import datetime
from collections import deque
from re import sub
import numpy as np

from pyworkforce.queuing.erlang import ErlangC

from datetime import datetime as dt

HMin = 60
DayH = 24
HMDELIMITER = '-'

def hh_mm(time_string, delimiter = ':'):
    hh = int(time_string.split(delimiter)[0])
    mm = int(time_string.split(delimiter)[1])

    return (hh, mm)


def hh_mm_time(time_string, offset = None, delimiter = ':') -> datetime.time:
    (hh, mm) = hh_mm(time_string, delimiter)

    if offset is not None:
        tz = datetime.timezone(datetime.timedelta(hours=offset))
        return datetime.time(hour=hh, minute=mm, tzinfo=tz)

    return datetime.time(hour=hh, minute=mm)


def hh_mm_timedelta(time_string, delimiter = ':') -> datetime.timedelta:
    (hh, mm) = hh_mm(time_string, delimiter)
    return datetime.timedelta(hours=hh, minutes=mm)


def get_start_from_shift_short_name(name):
    # 3_12_7_45
    start_time = dt.strptime(f'{name.split("_")[2]}:{name.split("_")[3]}',"%H:%M")
    return sub(".*\\s+", "", str(start_time))


def get_start_from_shift_short_name_mo(name):
    # 3_12_7_45
    h = int(name.split("_")[2])
    m = int(name.split("_")[3])
    return f"{h:02}:{m:02}"


def get_duration_from_shift_short_name(name):
    # 3_12_7_45
    duration = int(name.split('_')[1])
    return duration


def get_shift_short_name(t, utc):
    duration = dt.strptime(t['duration'], "%H:%M").hour
    start = t['scheduleTimeStart'].replace(':', HMDELIMITER)
    end = t['scheduleTimeEndStart'].replace(':', HMDELIMITER)
    stepTime = dt.strptime(t['stepTime'], "%H:%M").minute
    return f'x_{utc}_{duration}_{start}_{end}_{stepTime}'


def required_positions(call_volume: int, aht: int, interval: int, art: int, service_level: int) -> int:
  """
  Calculates the required number of resources to serve requests.
  It calculates 'raw' positions, without any shrinkage etc.

  Parameters
  ----------
  call_volume: int
    Call intensivity over time, count
  aht: int
    average handling time of a single call, seconds
  interval: int
    an interval to plan for, seconds
  art: int
    Average response time, seconds
  service_level: int
    required service level to achieve 0-100

  Returns
  -------
  The required number of resources.
  It returns the closest int number of resources to achieve the requested service level.

  E.g. if service level is defined as 80%, but required resource = 14,
  then it could the future service level will be 82%, not 80% as it was requested.
  """
  erlang = ErlangC(transactions=call_volume, aht=aht, interval=interval, asa=art, shrinkage=0.0)
  positions_requirements = erlang.required_positions(service_level=service_level / 100.0, max_occupancy=1.00)
  return int(positions_requirements['positions'])


def upscale_and_shift(a, time_scale, shift_right_pos):
  scaled = [val for val in a for _ in range(time_scale)]
  items = deque(scaled)
  items.rotate(shift_right_pos)
  return list(items)

def rotate(a, shift_right_pos):
  items = deque([val for val in a])
  items.rotate(shift_right_pos)
  return list(items)

def genereate_shifts_coverage(base_spec, name, horizon_in_hours, start_hour, start_min, end_hour, end_min, step_mins):
  if (start_hour == end_hour):
    slots = (end_min - start_min) // step_mins + 1
    res = {}
    for i in range(slots):
        s_name = f'{name}_{horizon_in_hours}_{start_hour}_{i * step_mins}'
        res[s_name] = rotate(base_spec, i)
    return res
  else:
    slots = ((end_hour * HMin + end_min) - (start_hour * HMin + start_min)) // step_mins + 1  # add 1 - to include end
    res = {}
    for i in range(slots):
        s_name = f'{name}_{horizon_in_hours}_{start_hour + (i * step_mins // HMin)}_{i * step_mins % HMin}'
        res[s_name] = rotate(base_spec, i)
    return res


def unwrap_shift(encoded_shift_name, with_breaks = False):
    t = ShiftSpec.decode_shift_spec(encoded_shift_name)
    return t.generate_coverage()


def all_zeros_shift():
    spec = [0 for i in range(DayH)]
    step_mins = 15  # todo
    scaled = upscale_and_shift(spec, HMin // step_mins, 0)

    return scaled


class ShiftSpec(object):
    def __init__(self, name, name_prefix, offset, start_start, start_end, duration_hours, step_minutes):
        self._name = name
        self._name_prefix = name_prefix
        self._offset = offset
        self._start_start = start_start
        self._start_end = start_end
        self._duration_hours = duration_hours
        self._step_minutes = step_minutes

    @property
    def name(self) -> str:
        return self._name

    @property
    def name_prefix(self) -> str:
        return self._name_prefix

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def start_start(self) -> datetime.time:
        return self._start_start

    @property
    def start_end(self) -> datetime.time:
        return self._start_end

    @property
    def duration_hours(self) -> int:
        return self._duration_hours

    @property
    def step_minutes(self) -> int:
        return self._step_minutes

    @staticmethod
    def decode_shift_spec(encoded_shift_name):  # x_3_9_06-00_12-45_15
        cx = encoded_shift_name.count('_')

        if cx == 3:  # 'x_9_6_0'
            name, duration, start_hour, start_min = encoded_shift_name.split('_')
            return ShiftSpec(name = encoded_shift_name, name_prefix=name,
                             offset = 0,
                             duration_hours=int(duration),
                             start_start=datetime.time(hour=int(start_hour), minute=int(start_min)),
                             start_end=None,
                             step_minutes=15
                             )
        elif cx == 4:
            name, duration, start, end, step = encoded_shift_name.split('_')
            return ShiftSpec(name=encoded_shift_name, name_prefix=name,
                             duration_hours=int(duration),
                             step_minutes=int(step),
                             start_start=hh_mm_time(start, delimiter='-'),
                             start_end=datetime.time(hour=int(end)))
        elif cx == 5:  # x_3_9_06-00_12-45_15
            name, utc, duration, start, end, step = encoded_shift_name.split('_')

            return ShiftSpec(name=encoded_shift_name, name_prefix=name,
                             offset=int(utc),
                             duration_hours=int(duration),
                             step_minutes=int(step),
                             start_start=hh_mm_time(start, offset=int(utc), delimiter='-'),
                             start_end=hh_mm_time(end, offset=int(utc), delimiter='-'))
        else:
            raise "Shift spec not supported"

    def generate_coverage(self) -> list:
        slots = DayH * HMin // self.step_minutes
        duration = self.duration_hours * HMin // self.step_minutes
        start_offset = (self.start_start.hour * HMin + self.start_start.minute) // self.step_minutes

        base_spec = [1 if (i < duration) else 0 for i in range(slots)]

        base_spec = deque(base_spec)
        base_spec.rotate(start_offset)
        base_spec = list(base_spec)

        return base_spec


def get_shift_coverage(shifts):
    shift_cover = {}
    for i in shifts:
        a: ShiftSpec = ShiftSpec.decode_shift_spec(i)
        base_spec = a.generate_coverage()

        res = genereate_shifts_coverage(base_spec, a.name_prefix, a.duration_hours,
                                        a.start_start.hour, a.start_start.minute,
                                        a.start_end.hour, a.start_end.minute,
                                        a.step_minutes)
        shift_cover = shift_cover | res

    return shift_cover


def get_shift_colors(shift_names):
    shift_colors = {}
    for i in shift_names:
        if "Morning" in i:
            shift_colors[i] = '#34eb46'
        else:
            shift_colors[i] = '#0800ff'
    return shift_colors


def count_consecutive_zeros(shift_or):
    previous = 0
    count = 1
    for c in shift_or:
        if previous == 0 and c == 0:
            count += 1
        previous = c
    return count


def get_12h_transitional_shifts(shift_names):
    res = []
    for i in shift_names:
        t = decode_shift_spec(i)
        if (t.start + t.duration > 24):
            res.append(i)
    return res


def build_non_sequential_shifts(shift_names, h_distance, m_step):
    transitional_shifts = get_12h_transitional_shifts(shift_names)
    exclude_transitional = [t for t in shift_names if t not in transitional_shifts]
    res = []
    for i in range(len(transitional_shifts)):
        name_o = transitional_shifts[i]
        shift_o = np.array(unwrap_shift(name_o))
        shift_o_first_zero_pos = min(np.where(shift_o == 0)[0])
        for j in range(len(exclude_transitional)):
            name_d = exclude_transitional[j]
            shift_d = np.array(unwrap_shift(name_d))
            shift_d_first_non_zero_pos = min(np.where(shift_d == 1)[0])
            distance = (shift_d_first_non_zero_pos - shift_o_first_zero_pos) / (1.0 * HMin / m_step)
            if (distance < h_distance):
                res.append({
                    "origin": name_o,
                    "destination":name_d
                })
    return res

class ShiftSchema:
    def __init__(self, shift_name, shift_id, schema_id, utc, min_start_time, max_start_time, duration_time, holidays_min, holidays_max, work_min, work_max, employee_count):
        self.shift_name = shift_name
        self.shift_id = shift_id
        self.schema_id = schema_id
        self.utc = utc
        self.min_start_time = min_start_time
        self.max_start_time = max_start_time
        self.duration_time = duration_time
        self.holidays_min = holidays_min
        self.holidays_max = holidays_max
        self.work_min = work_min
        self.work_max = work_max
        self.employee_count=employee_count