import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time
import yfinance as yf
import sqlite3
from datetime import datetime


def create_db():
    conn = sqlite3.connect('asset_prices.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS asset_prices (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL ,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            change FLOAT ,
            category TEXT NOT NULL,
            period TEXT ,
            timeframe TEXT NOT NULL,
            PRIMARY KEY (ticker, date, timeframe)
            )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS ticker_list (
            ticker TEXT NOT NULL,
            category TEXT NOT NULL,
            change FLOAT NOT NULL,
            close FLOAT NOT NULL,
            PRIMARY KEY (ticker)
            )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS favourites(
            ticker TEXT NOT NULL,
            category TEXT NOT NULL,
            change FLOAT NOT NULL,
            close FLOAT NOT NULL,
            PRIMARY KEY (ticker)
            )''')
    cursor.execute('''CREATE INDEX IF NOT EXISTS idx_ticker_date 
                         ON asset_prices(ticker, date)''')
    
    cursor.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    conn.close()
    return


def updateTickerListdata(ticker: str, category:str, change:float , close:float):
    ticker = ticker.upper()
    with sqlite3.connect("asset_prices.db") as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE ticker_list SET category = ?, change = ?, close = ? WHERE ticker = ?', (category, change, close, ticker))
        data=cursor.fetchall()    
    return data


def insertDataIntoTickerList(ticker :str ):
    ticker = ticker.upper()
    with sqlite3.connect("asset_prices.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
                       INSERT OR IGNORE INTO ticker_list (ticker, category, change, close) 
                       SELECT ticker, category, change, close 
                       FROM asset_prices
                       WHERE ticker = ?, ''', (ticker, ))
        data=cursor.fetchall()    
    return data


def read_newtickerlist():
    with sqlite3.connect('asset_prices.db') as conn :
        data=pd.read_sql_query("SELECT * FROM ticker_list", conn)
    data .dropna(inplace=True)
    print(data)
    return data



def deleteDataFromFavourites(ticker :str):
    ticker = ticker.upper()
    with sqlite3.connect("asset_prices.db") as conn:    
        cursor = conn.cursor()
        cursor.execute("DELETE FROM favourites WHERE ticker = ?", (ticker, ))
        conn.commit()
    return


def insertDataIntoFavourites(ticker :str):
    ticker = ticker.upper()
    with sqlite3.connect("asset_prices.db") as conn:    
        cursor = conn.cursor()
        cursor.execute('''INSERT OR IGNORE INTO favourites (ticker, category, change, close)
                       SELECT ticker, category, change, close 
                       FROM asset_prices
                       WHERE ticker = ?''', (ticker, ))
        conn.commit()
    return


def readFavorites():
    with sqlite3.connect("asset_prices.db") as conn :
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM favourites")
        data = cursor.fetchall()
    return data



def get_data(ticker :str, start_date:str = None, end_date:str = None,period :str = None,  timeframe : str = "1d") -> pd.DataFrame:
    if start_date and end_date:
        data = yf.download(ticker, start=start_date, end=end_date, threads=True, period=period, interval= timeframe, multi_level_index=False)[["Open", "High", "Low", "Close"]]
    elif timeframe:
        if timeframe.endswith("m"):
            data = yf.download(ticker, period="1d", interval= timeframe, multi_level_index=False)[["Open", "High", "Low", "Close"]]
        elif timeframe.endswith("h"):
            data = yf.download(ticker, period="1mo", interval= timeframe, multi_level_index=False)[["Open", "High", "Low", "Close"]]
        else : 
            data = yf.download(ticker, period="max", interval= timeframe, multi_level_index=False)[["Open", "High", "Low", "Close"]]
        
    data.reset_index(inplace=True)
    data.columns=data.columns.str.lower()    
    if 'datetime' in data.columns:
        data.rename(columns={"datetime": "date"}, inplace=True)
       
    data["ticker"] = ticker
    data["timeframe"] = timeframe
    data["period"]  = period
    data["change"]= data["close"].pct_change()
    if ticker.endswith(("-usd", "-USD")):
        data["category"] = "crypto"
    elif ticker.endswith(("=X", "=x")):
        data["category"] = "currency"
    else:
        data["category"] = "stock"
   
    return data


   
   
def read_ticker_list() :
    with sqlite3.connect('asset_prices.db') as conn :
        data = pd.read_sql_query("""
            SELECT category, ticker, close, change 
            FROM asset_prices 
            WHERE rowid IN (
                SELECT MAX(rowid) 
                FROM asset_prices 
                GROUP BY ticker
            )
        """, conn)
    data.dropna(inplace=True)   
    
    return data



def read_db(ticker:str, start_date: str = None , end_date: str = None, timeframe  : str = "1d") -> pd.DataFrame:
    ticker = ticker.upper()
    today=datetime.now().strftime("%Y-%m-%d")
    create_db()
    try :
        with sqlite3.connect('asset_prices.db') as conn :
            cursor = conn.cursor()
            
            query = cursor.execute("SELECT MAX(date) FROM asset_prices WHERE ticker = ? AND timeframe = ?", (ticker,timeframe))
            data=cursor.fetchall()
            if not data:
                get_data(ticker, start_date="2008-01-01", end_date=today, timeframe=timeframe)
                query = cursor.execute("SELECT * FROM asset_prices WHERE ticker = ?", (ticker,))
                data = cursor.fetchall()
                
            else:
                if start_date and end_date:
                    if data[-1][1][:10] != today:
                        get_data(ticker, start_date=data[-1][1][:10], end_date=today, timeframe=timeframe)
                        query = cursor.execute("SELECT * FROM asset_prices WHERE ticker = ?", (ticker,))
                        data = cursor.fetchall()
                    elif timeframe != data[-1][7]:
                        get_data(ticker, start_date=data[-1][1][:10], end_date=today, timeframe=timeframe)
                        query = cursor.execute("SELECT * FROM asset_prices WHERE ticker = ?", (ticker,))
                        data = cursor.fetchall()
                    else:
                        query = cursor.execute("SELECT * FROM asset_prices WHERE ticker = ? AND date BETWEEN ? AND ?", (ticker, start_date, end_date))
                        data = cursor.fetchall()
                else:
                    if data[-1][1][:10] != today:
                        get_data(ticker, start_date=data[-1][1][:10], end_date=today, timeframe=timeframe)
                        query = cursor.execute("SELECT * FROM asset_prices WHERE ticker = ?", (ticker,))
                        data = cursor.fetchall()
                    elif timeframe != data[-1][7]:
                        get_data(ticker, start_date=data[-1][1][:10], end_date=today, timeframe=timeframe)
                        query = cursor.execute("SELECT * FROM asset_prices WHERE ticker = ?", (ticker,))
                        data = cursor.fetchall()
                    else:
                        query = cursor.execute("SELECT * FROM asset_prices WHERE ticker = ?", (ticker,))
                        data = cursor.fetchall()
            
            data = pd.DataFrame(data, columns=[desc[0] for desc in cursor.description])
            data.drop_duplicates(inplace=True)
            data.dropna(inplace=True)
             
        return data
    except Exception as e:
        raise ValueError(f"Error reading database: {e}") 

#read_db("eth-usd", timeframe="1d")

def read_db_v2(ticker:str, start_date: str = None, end_date: str = None, period: str = None, timeframe: str = "1d") -> pd.DataFrame:
    ticker = ticker.upper()
    today = datetime.now().strftime("%Y-%m-%d")
    create_db()
    
    try:
        with sqlite3.connect('asset_prices.db') as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT MAX(date) FROM asset_prices WHERE ticker = ? AND timeframe = ?", (ticker, timeframe))
            result = cursor.fetchone()
            max_date = result[0] if result and result[0] else None  
            
            if not max_date and timeframe != "1d":
                updated_data =get_data(ticker ,timeframe=timeframe)
                if not updated_data.empty:
                    updated_data.to_sql("asset_prices", conn, if_exists="append", index=False)
                
            elif not max_date :
                updated_data =get_data(ticker ,start_date="2008-01-01", end_date=today,timeframe=timeframe)
                updated_data.to_sql("asset_prices", conn, if_exists="append", index=False)
                
            else:
                last_date = max_date[:10]  
                if last_date != today:
                    updated_data = get_data(ticker, start_date=last_date, end_date=today, timeframe=timeframe)
                    if not updated_data.empty:
                        updated_data['date'] = updated_data['date'].astype(str)
                        cursor.executemany(
                        """INSERT OR IGNORE INTO asset_prices 
                           (ticker, date, open, high, low, close, change,period,  category, timeframe) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        updated_data.values.tolist()
                        )
                        conn.commit()
            if period : 
                updated_data = get_data(ticker, start_date=None, end_date=None, period=period, timeframe=timeframe)
                if not updated_data.empty:
                    updated_data['date'] = updated_data['date'].astype(str)
                    cursor.executemany(
                        """INSERT OR IGNORE INTO asset_prices 
                           (ticker, date, open, high, low, close, change, period,category, timeframe) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        updated_data.values.tolist()
                    )
                    conn.commit()
                    
                
             
            if start_date and end_date:
                cursor.execute("SELECT * FROM asset_prices WHERE ticker = ? AND timeframe = ? AND date BETWEEN ? AND ?",
                              (ticker, timeframe, start_date, end_date))
            elif ticker and timeframe:
                cursor.execute("SELECT * FROM asset_prices WHERE ticker = ? AND timeframe = ?",
                              (ticker, timeframe))
            else:
                cursor.execute("SELECT * FROM asset_prices")
            
           
            
            data = cursor.fetchall()
            data = pd.DataFrame(data, columns=[desc[0] for desc in cursor.description])
            data.drop_duplicates(inplace=True)
            critical_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'category', 'timeframe']
            data.dropna(subset=critical_columns, inplace=True)
            print(data.head(3))
            return data
            
    except Exception as e:
        raise ValueError(f"Error reading database: {e}")


#read_db_v2("nvda", timeframe="15m")






def calculate_query_return(ticker: str, start_date: str, end_date: str) -> float:
    try :
        query = read_db_v2(ticker, start_date , end_date)
        query_return = (query.close.iat[-1]/ query.close.iat[0]) - 1

        if len(query) <2 :
            raise ValueError("Error : query length is less than 2")

    except Exception as e:
        raise ValueError(f"Error reading database: {e}")
        
    return query_return




def to_float(x):
    """Convert NumPy arrays or scalars safely to a Python float or list of floats."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    if isinstance(x, (np.ndarray, np.generic)):
        if x.ndim == 0:
            return float(x.item())             
        elif x.ndim == 1:
            return [float(v) for v in x]        
        else:
            return float(np.mean(x))            
    return float(x)


def optimize_calc(ticker: str , start_date: str,end_date: str) -> tuple:

    """

    Find pattern into the time series

    Usage:

    start: start index of the time series

    end: end index of the time series

    """

    'using sliding window instead of nested for loop'

    start_time = time.time()
    
    data = read_db(ticker)
    
    data["date"] = pd.to_datetime(data["date"])  
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    mask = (data["date"] >= start_date) & (data["date"] <= end_date)
    array = data.loc[mask, "close"].values

    array2 = data["close"].values


    m = len(array)

    n = len(array2)


    subsequences = sliding_window_view(array2, m) 
    distances = np.sum(np.abs(subsequences - array), axis=1)

    best_start = np.argmin(distances)
    best_distance = distances[best_start]
    best_indices = list(range(best_start, best_start + m))
    best_subarray = array2[best_start:best_start + m]

    best_dates = data["date"].iloc[best_indices].tolist()

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

    return best_indices, best_dates, best_subarray, array, array2, data , best_distance


#optimize_calc("btc-usd", "2025-01-10", "2025-01-20")



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

    if len(array) == 0:
        raise ValueError("Query array is empty")
    if len(array2) == 0:
        raise ValueError("Reference array is empty")
    if len(array2) < len(array):
        raise ValueError(f"Reference array (len={len(array2)}) is shorter than query array (len={len(array)})")
    if len(dates) != len(array2):
        raise ValueError(f"Dates array (len={len(dates)}) must match reference array (len={len(array2)})")

    m = len(array)
    n = len(array2)
    
    array_mean = np.mean(array)
    array_std = np.std(array)
    if array_std > 0:
        array_normalized = (array - array_mean) / array_std
    else:
        array_normalized = array - array_mean

    if wrap:
        extended_prices = np.concatenate([array2, array2[:m-1]])
        extended_dates = np.concatenate([dates, dates[:m-1]])
        source_array = extended_prices
    else:
        extended_prices = array2
        extended_dates = dates
        source_array = array2
    
    subsequences = sliding_window_view(extended_prices, m)
    
    subseq_means = np.mean(subsequences, axis=1, keepdims=True)
    subseq_stds = np.std(subsequences, axis=1, keepdims=True)
    
    subseq_stds = np.where(subseq_stds > 0, subseq_stds, 1.0)
    subsequences_normalized = (subsequences - subseq_means) / subseq_stds
    
    if metric == "l1":
        dists = np.sum(np.abs(subsequences_normalized - array_normalized), axis=1)
    elif metric == "l2":
        dists = np.sqrt(np.sum((subsequences_normalized - array_normalized) ** 2, axis=1))
    else:
        raise ValueError("metric must be 'l1' or 'l2'")
    
    k_actual = min(k, len(dists))
    
    if k_actual > 0:
        best_idx = np.argpartition(dists, k_actual-1)[:k_actual]
        best_idx = best_idx[np.argsort(dists[best_idx])]
    else:
        best_idx = np.array([], dtype=int)
    
    best_distances = dists[best_idx].tolist()
    best_starts = best_idx.tolist()
    best_indices = [list(range(start, start + m)) for start in best_starts]
    best_subarrays = [source_array[start:start+m].tolist() for start in best_starts]
    best_dates = [extended_dates[start:start+m].tolist() for start in best_starts]
    
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
            source_array = extended
            subsequences = sliding_window_view(extended, window_size)
            if dates is not None:
                extended_dates = np.concatenate([dates, dates[:window_size - 1]])
        else:
            source_array = array2
            subsequences = sliding_window_view(array2, window_size)
            if dates is not None:
                extended_dates = dates
        
        dates_to_slice = extended_dates if dates is not None else None

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
            
            prices = source_array[start:start+window_size] 
            
            current_dates = dates_to_slice[start:start+window_size] if dates_to_slice is not None else None
            
            best_matches.append((
                start,
                dist,
                window_size,
                prices,
                current_dates
            ))

    best_matches = sorted(best_matches, key=lambda x: x[1])[:k]

    best_indices = [list(range(start, start + w)) for start, _, w, _, _ in best_matches]
    best_distances = [dist for _, dist, _, _, _ in best_matches]
    best_subarrays = [sub for _, _, _, sub, _ in best_matches]
    best_dates = [d for _, _, _, _, d in best_matches]

    elapsed_time = time.time() - start_time
    print(f"Elapsed time (variable-length search): {elapsed_time:.6f} seconds")

    return best_indices, best_dates, best_subarrays, best_distances, array, array2
