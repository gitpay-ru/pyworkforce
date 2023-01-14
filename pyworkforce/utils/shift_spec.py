from collections import deque
import numpy as np

from pyworkforce.queuing.erlang import ErlangC

from datetime import datetime as dt

HMin = 60
DayH = 24

def get_shift_short_name(t, utc):
    duration = dt.strptime(t['duration'], "%H:%M").hour
    start = dt.strptime(t['scheduleTimeStart'], "%H:%M").hour
    end = dt.strptime(t['scheduleTimeEndStart'], "%H:%M").hour
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
  return positions_requirements['positions']

def upscale_and_shift(a, time_scale, shift_right_pos):
  scaled = [val for val in a for _ in range(time_scale)]
  items = deque(scaled)
  items.rotate(shift_right_pos)
  return list(items)

def genereate_shifts_coverage(shift_hours, name, horizon_in_hours, start_hour, end_hour, step_mins):
  time_scale = int(HMin / step_mins)
  slots = time_scale * (end_hour - start_hour)
  res = {}
  for i in range(slots):
    s_name = f'{name}_{horizon_in_hours}_{start_hour + (i * step_mins // HMin)}_{i * step_mins % HMin}'
    res[s_name] = upscale_and_shift(shift_hours, time_scale, i) 
  return res

def unwrap_shift(encoded_shift_name, with_breaks = False):
    t = decode_shift_spec(encoded_shift_name)
    if (with_breaks):
        base_spec = [1 if (i < t.duration and i != t.duration // 2) else 0 for i in range(DayH)]
    else:
        base_spec = [1 if (i < t.duration) else 0 for i in range(DayH)]
    base_spec = deque(base_spec)
    base_spec.rotate(t.start)
    base_spec = list(base_spec)

    step_mins = 15 #todo
    scaled = upscale_and_shift(base_spec, HMin // step_mins,t.offset // step_mins) 
    return scaled

def decode_shift_spec(encoded_shift_name):
    cx = encoded_shift_name.count('_')
    class Object(object):
        def __str__(self):
            return "todo: replace tostring()"
    t = Object()
    if cx == 3:
        name, duration, start, offset = encoded_shift_name.split('_')
        t.offset = int(offset)
    elif cx == 4:
        name, duration, start, end, step = encoded_shift_name.split('_')
        t.end = int(end)
        t.step = int(step)
    elif cx == 5:
        utc, name, duration, start, end, step = encoded_shift_name.split('_')
        t.end = int(end)
        t.step = int(step)
    else:
        raise "Shift spec not supported"

    t.name = name
    t.duration = int(duration)
    t.start = int(start)
    return t

def get_shift_coverage(shifts, with_breaks = False):
    shift_cover = {}
    for i in shifts:
        a = decode_shift_spec(i)
        if (with_breaks):
            base_spec = [1 if (i < a.duration and i != a.duration // 2) else 0 for i in range(DayH)]
        else:
            base_spec = [1 if (i < a.duration) else 0 for i in range(DayH)]
        base_spec = deque(base_spec)
        base_spec.rotate(a.start)
        base_spec = list(base_spec)
        res = genereate_shifts_coverage(base_spec, a.name, a.duration, a.start, a.end, a.step)
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

