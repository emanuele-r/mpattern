import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time
import yfinance as yf
import sqlite3
from datetime import datetime, timedelta
import subprocess
import asyncio
import requests
import os
from dotenv import load_dotenv


def create_db():
    conn = sqlite3.connect("asset_prices.db")
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS asset_prices (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,          
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            change REAL,
            category TEXT NOT NULL,
            period TEXT,
            timeframe TEXT NOT NULL,
            FOREIGN KEY (ticker) REFERENCES symbols(ticker)
                ON DELETE CASCADE ON UPDATE CASCADE,
            PRIMARY KEY (ticker, date, timeframe)
        ) WITHOUT ROWID;
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ticker_list (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            change REAL NOT NULL,
            close REAL NOT NULL,
            FOREIGN KEY (ticker) REFERENCES symbols(ticker)
                ON DELETE CASCADE ON UPDATE CASCADE
        );
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS favourites (
            ticker TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            change REAL NOT NULL,
            close REAL NOT NULL,
            FOREIGN KEY (ticker) REFERENCES symbols(ticker)
                ON DELETE CASCADE ON UPDATE CASCADE
        ) WITHOUT ROWID;
    """
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS symbols(
            ticker TEXT PRIMARY KEY
    )"""
    )

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_asset_category ON asset_prices (category);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_asset_ticker_date ON asset_prices (ticker, date DESC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_ticker_category ON ticker_list (category);"
    )

    cursor.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    conn.close()
    return


create_db()


def deleteDataFromAssetPrices():
    with sqlite3.connect("asset_prices.db") as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM asset_prices WHERE ticker >'AAB%' ")
        conn.commit()
    return


def checkTickerExistence():
    subprocess.run(["./update_ticker.sh"], shell=True)
    print("tickers updated")
    return


def readCategory():
    with sqlite3.connect("asset_prices.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT category FROM ticker_list GROUP BY category")
        data = cursor.fetchall()
    return data


def readTickerList(category: str = None):
    with sqlite3.connect("asset_prices.db") as conn:
        cursor = conn.cursor()
        if category:
            cursor.execute(
                """
                SELECT  ticker, category, change, close
                FROM ticker_list
                WHERE category = ?
            """,
                (category,),
            )
            data = cursor.fetchall()

            return data
        else:
            cursor.execute(
                """
           SELECT 
    t.ticker,
    t.category,
    t.change,
    t.close,
    a.price
FROM ticker_list AS t
JOIN asset_prices AS a 
    ON t.ticker = a.ticker
JOIN (
    SELECT ticker, MAX(date) AS max_date
    FROM asset_prices
    GROUP BY ticker
) AS latest
    ON a.ticker = latest.ticker
   AND a.date = latest.max_date;

        """,
            )

        data = cursor.fetchall()

    return data


def insertDataIntoTickerList():

    with sqlite3.connect("asset_prices.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        INSERT INTO ticker_list (ticker, category, change, close)
        SELECT ap.ticker, ap.category, ap.change, ap.close
        FROM asset_prices ap
        WHERE ap.date = (
        SELECT MAX(a2.date)
        FROM asset_prices a2
        WHERE a2.ticker = ap.ticker
        )
        ON CONFLICT(ticker) DO UPDATE SET
        category = excluded.category,
        change = excluded.change,
        close = excluded.close;
        """
        )
        conn.commit()
        return cursor.rowcount


def getNews(query: str, lang: str = "en") -> dict:
    load_dotenv("/home/emanuelerossi/dev/mpattern/news_api.env")
    api_key = os.getenv("NEWS_API_KEY")
    url = f"https://newsdata.io/api/1/latest?apikey={api_key}&q={query}&language={lang}"
    response = requests.get(url).json()["results"]

    news = []
    for row in response:
        news.append(
            {
                "title": row["title"],
                "description": row["description"],
                "source": row["source_url"],
            }
        )
    return news


def deleteDataFromFavourites(ticker: str):
    ticker = ticker.upper()
    with sqlite3.connect("asset_prices.db") as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM favourites WHERE ticker = ?", (ticker,))
        conn.commit()
    return


def insertDataIntoFavourites(ticker: str):
    ticker = ticker.upper()
    with sqlite3.connect("asset_prices.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR IGNORE INTO favourites (ticker, category, change, close)
                       SELECT ticker, category, change, close 
                       FROM asset_prices
                       WHERE ticker = ?""",
            (ticker,),
        )
        conn.commit()
    return


def readFavorites():
    with sqlite3.connect("asset_prices.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM favourites")
        data = cursor.fetchall()
    return data


def get_data(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    period: str = None,
    timeframe: str = "1d",
) -> pd.DataFrame:
    if start_date and end_date:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            threads=True,
            period=period,
            interval=timeframe,
            multi_level_index=False,
        )[["Open", "High", "Low", "Close"]]
    elif timeframe:
        if timeframe.endswith("m"):
            data = yf.download(
                ticker,
                period="1d",
                interval=timeframe,
                threads=True,
                multi_level_index=False,
            )[["Open", "High", "Low", "Close"]]
        elif timeframe.endswith("h"):
            data = yf.download(
                ticker,
                period="1mo",
                interval=timeframe,
                threads=True,
                multi_level_index=False,
            )[["Open", "High", "Low", "Close"]]
        else:
            data = yf.download(
                ticker,
                period="max",
                interval=timeframe,
                threads=True,
                multi_level_index=False,
            )[["Open", "High", "Low", "Close"]]

    data.reset_index(inplace=True)
    data.columns = data.columns.str.lower()
    if "datetime" in data.columns:
        data.rename(columns={"datetime": "date"}, inplace=True)

    data["ticker"] = ticker
    data["timeframe"] = timeframe
    data["period"] = period
    data["change"] = data["close"].pct_change()
    if ticker.endswith(("-usd", "-USD")):
        data["category"] = "crypto"
    elif ticker.endswith(("=X", "=x")):
        data["category"] = "currency"
    else:
        data["category"] = "stock"

    return data


def popoulateDb():
    with sqlite3.connect("asset_prices.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker FROM symbols")
        data = cursor.fetchall()
        for row in data:
            ticker = row[0]
            print(f"downloading data for {ticker}")
            downloaded_data = get_data(ticker)

            if downloaded_data is not None and not downloaded_data.empty:
                downloaded_data.to_sql(
                    "asset_prices",
                    conn,
                    if_exists="append",
                    index=False,
                )
                print(f"✓ {ticker} downloaded")
            else:
                print("✗ No data returned")

    return


def read_db_v2(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    period: str = None,
    timeframe: str = "1d",
) -> pd.DataFrame:
    ticker = ticker.upper()
    today = (
        datetime.now().strftime("%Y-%m-%d")
        if timeframe == "1d"
        else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    try:
        with sqlite3.connect("asset_prices.db") as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT MAX (date) FROM asset_prices where ticker =? AND timeframe =?",
                (ticker, timeframe),
            )
            last_close = cursor.execute(
                "SELECT close from asset_prices where ticker = ? AND timeframe = 1m",
                (ticker,),
            )
            result = cursor.fetchone()
            isUptoDate = result[0] if result[0] is not None else None

            if isUptoDate != today and timeframe != "1d":
                upDateData = get_data(ticker=ticker, timeframe=timeframe)
                if not upDateData.empty:
                    records = [
                        (
                            ticker,
                            str(row["date"]),
                            row["open"],
                            row["high"],
                            row["low"],
                            row["close"],
                            row["change"],
                            row["category"],
                            row["period"],
                            timeframe,
                        )
                        for _, row in upDateData.iterrows()
                    ]

                    cursor.executemany(
                        "INSERT OR REPLACE INTO asset_prices (ticker, date, open, high, low, close,change,category, period, timeframe) VALUES (?, ?, ?, ?, ?, ?, ?, ?,?, ?)",
                        records,
                    )
            elif isUptoDate != today:
                upDateData = get_data(
                    ticker=ticker,
                    start_date="2008-01-01",
                    end_date=today,
                    timeframe=timeframe,
                )
                if not upDateData.empty:
                    records = [
                        (
                            ticker,
                            str(row["date"]),
                            row["open"],
                            row["high"],
                            row["low"],
                            row["close"] if row["close"] == last_close else last_close,
                            row["change"],
                            row["category"],
                            row["period"],
                            timeframe,
                        )
                        for _, row in upDateData.iterrows()
                    ]

                    cursor.executemany(
                        "INSERT OR REPLACE INTO asset_prices (ticker, date, open, high, low, close,change,category, period, timeframe) VALUES (?, ?, ?, ?, ?, ?, ?, ?,?, ?)",
                        records,
                    )
            if start_date and end_date:
                cursor.execute(
                    "SELECT * FROM asset_prices WHERE date BETWEEN ? AND ? AND ticker = ? AND timeframe = ?",
                    (start_date, end_date, ticker, timeframe),
                )
            elif ticker and timeframe:
                cursor.execute(
                    "SELECT * FROM asset_prices WHERE ticker = ? AND timeframe = ?",
                    (ticker, timeframe),
                )
            else:
                cursor.execute("SELECT * FROM asset_prices WHERE ticker = ?", (ticker,))

            updated_data = cursor.fetchall()
            updated_data = pd.DataFrame(
                updated_data, columns=[col[0] for col in cursor.description]
            )

    except Exception as e:
        raise ValueError(f"Error reading database : {e}")

    return updated_data


def calculate_query_return(ticker: str, start_date: str, end_date: str) -> float:
    try:
        query = read_db_v2(ticker, start_date, end_date)
        query_return = (query.close.iat[-1] / query.close.iat[0]) - 1

        if len(query) < 2:
            raise ValueError("Error : query length is less than 2")

    except Exception as e:
        raise ValueError(f"Error reading database: {e}")

    return query_return


def array_with_shift(
    array, array2, dates, shift_range: int = 0, k: int = 3, metric="l1", wrap=True
):
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
        raise ValueError(
            f"Reference array (len={len(array2)}) is shorter than query array (len={len(array)})"
        )
    if len(dates) != len(array2):
        raise ValueError(
            f"Dates array (len={len(dates)}) must match reference array (len={len(array2)})"
        )

    m = len(array)
    n = len(array2)

    array_mean = np.mean(array)
    array_std = np.std(array)
    if array_std > 0:
        array_normalized = (array - array_mean) / array_std
    else:
        array_normalized = array - array_mean

    if wrap:
        extended_prices = np.concatenate([array2, array2[: m - 1]])
        extended_dates = np.concatenate([dates, dates[: m - 1]])
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
        dists = np.sqrt(
            np.sum((subsequences_normalized - array_normalized) ** 2, axis=1)
        )
    else:
        raise ValueError("metric must be 'l1' or 'l2'")

    k_actual = min(k, len(dists))

    if k_actual > 0:
        best_idx = np.argpartition(dists, k_actual - 1)[:k_actual]
        best_idx = best_idx[np.argsort(dists[best_idx])]
    else:
        best_idx = np.array([], dtype=int)

    best_distances = dists[best_idx].tolist()
    best_starts = best_idx.tolist()
    best_indices = [list(range(start, start + m)) for start in best_starts]
    best_subarrays = [source_array[start : start + m].tolist() for start in best_starts]
    best_dates = [extended_dates[start : start + m].tolist() for start in best_starts]

    return best_indices, best_dates, best_subarrays, best_distances, array, array2


def pattern_forward_return(
    ticker: str,
    best_dates: list[list[str]],
    rolling_window_low: int = 7,
    rolling_window_high: int = 30,
) -> float:
    ticker = ticker.upper()

    with sqlite3.connect("asset_prices.db") as conn:
        cursor_monthly = conn.cursor()
        cursor_week = conn.cursor()
        cursor = conn.cursor()
        forward_return = []
        avgReturn = []
        summary = []
        seven_day_returns = []
        monthly_returns = []
        pattern_returns = []

        for idx, pat in enumerate(best_dates):
            date_str = pat[-1].strip()
            end_date_dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            end_week_dt = end_date_dt + timedelta(days=rolling_window_low)
            end_month_dt = end_date_dt + timedelta(days=rolling_window_high)
            cursor.execute(
                "SELECT close, date from asset_prices WHERE  date >= ? AND date < ? AND ticker = ?",
                (best_dates[idx][0], best_dates[idx][-1], ticker),
            )
            cursor_week.execute(
                """SELECT close, date from asset_prices WHERE date BETWEEN ? AND ? AND ticker = ?""",
                (end_date_dt, end_week_dt, ticker),
            )
            cursor_monthly.execute(
                """SELECT close from asset_prices WHERE date BETWEEN ? AND ? AND ticker = ?""",
                (end_date_dt, end_month_dt, ticker),
            )
            weekly_data = cursor_week.fetchall()
            monthly_data = cursor_monthly.fetchall()
            data = cursor.fetchall()
            sevenDayReturnAfterPattern = (
                weekly_data[-1][0] / weekly_data[0][0] - 1
            ) * 100
            monthlyReturnAfterPattern = (
                monthly_data[-1][0] / monthly_data[0][0] - 1
            ) * 100
            avgReturnForPattern = (data[-1][0] / data[0][0] - 1) * 100

            seven_day_returns.append(sevenDayReturnAfterPattern)
            monthly_returns.append(monthlyReturnAfterPattern)
            pattern_returns.append(avgReturnForPattern)

            forward_return.append(
                {
                    "patternIdx": idx,
                    "sevenDayReturnAfterPattern": sevenDayReturnAfterPattern,
                    "monthlyReturnAfterPattern": monthlyReturnAfterPattern,
                }
            )

        avgReturn.append(
            {
                "avgReturn": (
                    sum(pattern_returns) / len(pattern_returns)
                    if pattern_returns
                    else 0
                ),
                "avgSevenDayReturn": (
                    sum(seven_day_returns) / len(seven_day_returns)
                    if seven_day_returns
                    else 0
                ),
                "avgMonthlyReturn": (
                    sum(monthly_returns) / len(monthly_returns)
                    if monthly_returns
                    else 0
                ),
            }
        )

        summary.append(
            {
                "Pattern": forward_return,
                "summary": {
                    "Average 7 day return": avgReturn[0]["avgSevenDayReturn"],
                    "Average 30 day return": avgReturn[0]["avgMonthlyReturn"],
                    "Pattern Average return": avgReturn[0]["avgReturn"],
                },
            }
        )

    return summary


def dynamic_time_warping(
    array,
    array2,
    dates=None,
    shift_range: int = 0,
    k: int = 3,
    metric="l1",
    wrap=True,
    length_tolerance: int = 0,
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
            extended = np.concatenate([array2, array2[: window_size - 1]])
            source_array = extended
            subsequences = sliding_window_view(extended, window_size)
            if dates is not None:
                extended_dates = np.concatenate([dates, dates[: window_size - 1]])
        else:
            source_array = array2
            subsequences = sliding_window_view(array2, window_size)
            if dates is not None:
                extended_dates = dates

        dates_to_slice = extended_dates if dates is not None else None

        if window_size != m:
            query_rescaled = np.interp(
                np.linspace(0, m - 1, window_size), np.arange(m), array
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

            prices = source_array[start : start + window_size]

            current_dates = (
                dates_to_slice[start : start + window_size]
                if dates_to_slice is not None
                else None
            )

            best_matches.append((start, dist, window_size, prices, current_dates))

    best_matches = sorted(best_matches, key=lambda x: x[1])[:k]

    best_indices = [list(range(start, start + w)) for start, _, w, _, _ in best_matches]
    best_distances = [dist for _, dist, _, _, _ in best_matches]
    best_subarrays = [sub for _, _, _, sub, _ in best_matches]
    best_dates = [d for _, _, _, _, d in best_matches]

    elapsed_time = time.time() - start_time
    print(f"Elapsed time (variable-length search): {elapsed_time:.6f} seconds")

    return best_indices, best_dates, best_subarrays, best_distances, array, array2
