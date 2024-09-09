from typing import List, Tuple
import os
import psutil
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typeguard import typechecked


def get_memory_usage(scale: int = 3):
    """
    Returns the memory usage of the current process 

    Parameters
    ----------
    scale : int, optional
        The scale in which the memory usage is returned (Byte, KB, MB, GB).
        Encoded via the exponent of 1024. The default is 3, which returns the
        memory usage in GB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** scale)


def flatten_dict(d: dict, parent_key="", sep='.', skip_none=True) -> dict:
    """
    Flattens a dictionary of dictionaries recursively into a single dictionary with keys in the form of a.b.c

    Parameters
    ----------
    d : dict
        The dictionary to flatten.
    parent_key : str, optional
        The parent key. The default is "".
    sep : str, optional
        The separator to use, when combining the keys. The default is '.'.
    skip_none : bool, optional
        Whether to skip keys with a value of None. The default is True.
    Returns
    -------
    dict
        The flattened dictionary.
    """
    parent_key = str(parent_key)
    items = []
    for k, v in d.items():
        k = str(k)
        if skip_none and v is None:
            continue
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict) or hasattr(v, 'items'):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, ','.join([str(x) for x in v])))
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep='.') -> dict:
    """
    Unflatten a dictionary with keys in the form of a.b.c into a dictionary of dictionaries recursively

    Parameters
    ----------
    d : dict
        The dictionary to unflatten.
    sep : str, optional
        The separator to use, when combining the keys. The default is '.'.

    Returns
    -------
    dict
        The unflattened dictionary.
    """

    result_dict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = result_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result_dict


def segment_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    This function segments a given `mask` numpy array into consecutively chunks of 0s and 1s.
    Each segment is represented by a tuple containing the start and end indices of the segment.

    Args:
        mask (np.ndarray): a numpy array of 0s and 1s indicating a correct or incorrect prediction.

    Returns:
        List[Tuple]: List of tuples where each tuple contains start and end index of consecutively correct or incorrect predictions.

    Example:
        mask = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0])
        segment_mask(mask)
        
        Output: [(0, 1), (2, 3), (4, 5), (6, 6), (7, 7), (8, 10)]

    """
    segments = []
    start_index = None
    current_value = None
    for i, value in enumerate(mask):
        if start_index is None:
            start_index = i
            current_value = value
        elif value != current_value:
            end_index = i - 1
            segments.append((start_index, end_index))
            start_index = i
            current_value = value
    if start_index is not None:
        end_index = len(mask) - 1
        segments.append((start_index, end_index))
    return segments


def test_segment_mask():
    # Test case 1: all correct predictions
    mask = np.array([1, 1, 1, 1])
    expected_output = [(0, 3)]
    assert segment_mask(mask) == expected_output

    # Test case 2: all incorrect predictions
    mask = np.array([0, 0, 0, 0])
    expected_output = [(0, 3)]
    assert segment_mask(mask) == expected_output

    # Test case 3: alternating correct and incorrect predictions
    mask = np.array([1, 0, 1, 0])
    expected_output = [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert segment_mask(mask) == expected_output

    # Test case 4: multiple consecutive correct and incorrect predictions
    mask = np.array([1, 1, 0, 0, 1, 1, 0, 1])
    expected_output = [(0, 1), (2, 3), (4, 5), (6, 6), (7, 7)]
    assert segment_mask(mask) == expected_output

    # Test case 5: starting with correct predictions, ending with incorrect predictions and multiple consecutive correct and incorrect predictions of varying sequence lengths
    mask = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0])
    expected_output = [(0, 1), (2, 3), (4, 5), (6, 6), (7, 7), (8, 10)]
    assert segment_mask(mask) == expected_output


@typechecked
def save_image_html(figure: go.Figure, path: Path, **kwargs):
    """
    Save a plotly figure as an image and as an html file
    """

    figure.write_image(str(path.with_suffix(".png")), **kwargs)
    figure.write_html(str(path.with_suffix(".html")))


if __name__ == '__main__':
    test_segment_mask()
