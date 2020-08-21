from tensorboard.backend.event_processing import event_multiplexer, event_accumulator
import pandas as pd
from collections import defaultdict

def get_scalars(path, tag):
    """Get scalar data from a TB event file in DataFrame format

    Args:
        path: A file path to a directory containing tf events files, or a single tf events file. 
        tag: Tag for the scalar
    Returns:
        A pandas DataFrame object. Columns will be [step, wall_time, value]
    """
    accumulator = event_accumulator.EventAccumulator(path)
    accumulator.Reload()
    try:
        scalar_events = accumulator.Scalars(tag)
    except KeyError:
        raise KeyError('Tag {} not found. Available tags: {}'.format(tag, accumulator.Tags()['scalars']))
    return _parse_scalar_event_list(scalar_events)

def _parse_scalar_event_list(scalar_event_list):
    """Parse a list of TB ScalarEvent object to pandas data frame

    Args:
        scalar_event_list: a list of ScalarEvent objects
    """
    data = defaultdict(list)
    for scalar_event in  scalar_event_list:
        data['step'].append(scalar_event.step)
        data['wall_time'].append(scalar_event.wall_time)
        data['value'].append(scalar_event.value)
    df = pd.DataFrame(data)

    return df

