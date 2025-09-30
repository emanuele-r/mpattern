import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time
import yfinance as yf


def get_data(ticker, start_date, end_date, interval) -> pd.DataFrame:
    data = yf.download(ticker, start=start_date, end=end_date, interval= interval, multi_level_index=False)
    data.to_csv(f"{ticker}1D.csv")
    return data



def process_data(ticker : str ="EURUSD")  :
    data = pd.read_csv(f"{ticker}1D.csv", usecols=["Date", "Close"])
        
    return data[["Date", "Close"]]




def optimize_calc(ticker: str , start_date: str,end_date: str) -> tuple:

    """

    Find pattern into the time series

    Usage:

    start: start index of the time series

    end: end index of the time series

    """

    'using sliding window instead of nested for loop'

    start_time = time.time()
    
    data = process_data(ticker)
    data["Date"] = pd.to_datetime(data["Date"])  
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    mask = (data["Date"] >= start_date) & (data["Date"] <= end_date)
    array = data.loc[mask, "Close"].values

    array2 = data["Close"].values


    m = len(array)

    n = len(array2)


    subsequences = sliding_window_view(array2, m) 
    distances = np.sum(np.abs(subsequences - array), axis=1)

    best_start = np.argmin(distances)
    best_distance = distances[best_start]
    best_indices = list(range(best_start, best_start + m))
    best_subarray = array2[best_start:best_start + m]

    best_dates = data["Date"].iloc[best_indices].tolist()

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

    return best_indices, best_dates, best_subarray, array, array2, data , best_distance




def array_with_shift(array, array2, dates, shift_range: int = 0, k: int = 3, metric="l1", wrap=True):
    
    """
    Vectorized motif search: find top-k matches of `array` inside `array2`.
    
    Parameters
    ----------
    array : array-like
        The query array (length m).
    array2 : array-like
        The reference array (length n).
    shift_range : int, optional
        Maximum shift allowed between query and reference arrays.
    k : int, optional
        Best results to return.
    metric : {"l1", "l2"}
        Distance metric.
    wrap : bool
        Whether to allow wrapping (circular search).
    """
    array = np.asarray(array, dtype=float)
    array2 = np.asarray(array2, dtype=float)

    if len(array2) < len(array):
        return None, None, None, None, array, array2

    m = len(array)
    n = len(array2)

    if wrap:
        extended = np.concatenate([array2, array2[:m-1]])  
        subsequences = sliding_window_view(extended, m)
        extended_dates = np.concatenate([dates, dates[:m-1]])
    else:
        subsequences = sliding_window_view(array2, m)
        extended_dates = dates

    if metric == "l1":
        dists = np.sum(np.abs(subsequences - array), axis=1)
    elif metric == "l2":
        dists = np.sum((subsequences - array) ** 2, axis=1)
    else:
        raise ValueError("metric must be 'l1' or 'l2'")

    best_idx = np.argpartition(dists, k)[:k]  
    best_idx = best_idx[np.argsort(dists[best_idx])]

    best_distances = dists[best_idx].tolist()
    best_starts = best_idx.tolist()
    best_indices = [list(range(start, start + m)) for start in best_starts]
    best_subarrays = [array2[start:start+m] for start in best_starts]
    best_dates = [extended_dates[start:start+m] for start in best_starts]

    return best_indices, best_dates, best_subarrays, best_distances, array, array2



def dynamic_time_warping(
    array,
    array2,
    dates=None,
    shift_range: int = 0,
    k: int = 3,
    metric="l1",
    wrap=True,
    length_tolerance: int = 0
):
    """
    Vectorized motif search: find top-k matches of `array` inside `array2`,
    allowing for variable-length motifs.

    Parameters
    ----------
    array : array-like
        The query array (length m).
    array2 : array-like
        The reference array.
    dates : array-like, optional
        Dates aligned with array2.
    shift_range : int, optional
        (Unused for now) Maximum shift allowed between query and reference arrays.
    k : int, optional
        Best results to return.
    metric : {"l1", "l2"}
        Distance metric.
    wrap : bool
        Whether to allow wrapping (circular search).
    length_tolerance : int, optional
        Allowed variation in subsequence length around len(array).
        e.g., 2 means [m-2 ... m+2] window sizes will be checked.
    """

    array = np.asarray(array, dtype=float)
    array2 = np.asarray(array2, dtype=float)
    if dates is not None:
        dates = pd.to_datetime(dates)  # ensure datetime

    if len(array2) < len(array):
        return None, None, None, None, array, array2

    m = len(array)
    n = len(array2)

    start_time = time.time()

    best_matches = []

    for window_size in range(max(2, m - length_tolerance), m + length_tolerance + 1):
        if window_size > n:
            continue

        if wrap:
            extended = np.concatenate([array2, array2[:window_size - 1]])
            subsequences = sliding_window_view(extended, window_size)
            if dates is not None:
                extended_dates = np.concatenate([dates, dates[:window_size - 1]])
        else:
            subsequences = sliding_window_view(array2, window_size)
            if dates is not None:
                extended_dates = dates

        if window_size != m:
            query_rescaled = np.interp(
                np.linspace(0, m - 1, window_size),
                np.arange(m),
                array
            )
        else:
            query_rescaled = array

        if metric == "l1":
            dists = np.sum(np.abs(subsequences - query_rescaled), axis=1)
        elif metric == "l2":
            dists = np.sum((subsequences - query_rescaled) ** 2, axis=1)
        else:
            raise ValueError("metric must be 'l1' or 'l2'")

        for start, dist in enumerate(dists):
            best_matches.append((
                start,
                dist,
                window_size,
                subsequences[start],
                extended_dates[start:start+window_size] if dates is not None else None
            ))

    best_matches = sorted(best_matches, key=lambda x: x[1])[:k]

    best_indices = [list(range(start, start + w)) for start, _, w, _, _ in best_matches]
    best_distances = [dist for _, dist, _, _, _ in best_matches]
    best_subarrays = [sub for _, _, _, sub, _ in best_matches]
    best_dates = [d for _, _, _, _, d in best_matches]

    elapsed_time = time.time() - start_time
    print(f"Elapsed time (variable-length search): {elapsed_time:.6f} seconds")

    return best_indices, best_dates, best_subarrays, best_distances, array, array2
