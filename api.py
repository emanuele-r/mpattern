from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from utils import *
from pydantic import BaseModel
from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, Response
from datetime import datetime
import time
import os


app = FastAPI(
    title="Time Series Motif API",
    description="API for time series analysis and motif discovery",
    version="1.0.0"
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   

class SubsequenceMatch(BaseModel):
    dates: List[str]
    closes: List[float]
    similarity: Union[float, List[float]]
    query_return: Union[float, List[float]]
    description : str


class SubsequenceResponse(BaseModel):
    matches: List[SubsequenceMatch]

class HistoricalPrice(BaseModel):
    date: str
    close: float


class HistoricalPricesResponse(BaseModel):
    prices: List[HistoricalPrice]






@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str = ""):
    origin = request.headers.get("origin", "*")
    acrh = request.headers.get("Access-Control-Request-Headers", "*")

    response = Response()
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = acrh
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

@app.get("/health")
def health_check():
    return {"status": "ok, faggot"}

@app.get("/get_favorites")
def get_favorites_ticker():
    try :
        data= readFavorites()
        
        
        categoryTypes = []
        tickers = []

        for row in data:
            ticker, category, change , close= row
            categoryTypes.append({"category": category, "ticker": ticker, "change": change , "close": close})
            tickers.append({"category": category, "symbol": ticker, "change": change, "close": close})

            
        
        favourites = {"categoryTypes" : categoryTypes ,  "tickers" : tickers}
        
            
        
        return favourites
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/update_favorites")
def addToFavourites(ticker: str = Query(..., description="Ticker symbol")):
    try:
        insertDataIntoFavourites(ticker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/delete_favorites')
def deleteFromFavourites(ticker: str = Query(..., description="Ticker symbol")):
    try:
        deleteDataFromFavourites(ticker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_ticker_list")
def get_tickers():
   
    try:
        data = readTickerList()  
        categoryTypes = []
        tickers = []
        for row in data:
            ticker, category, change, price, id = row  

            categoryTypes.append({"category": f"{id}-{category}", "ticker": ticker})
            tickers.append({"category": f"{id}-{category}", "symbol": ticker, "price": price, "change": change})

        prices = {"categoryTypes": categoryTypes, "tickers": tickers}
        print(prices)
        return prices
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/historical_prices")
def read_data(
    ticker: str = Query(..., description="Ticker symbol"), 
    start_date : str = Query(default=None, description="Start date interval (Optional)"),
    end_date : str= Query(default=None, description="End date interval(Optional)"),
    timeframe :str =Query(default="1d", description="Timeframe (Optional)"),
    ):
    """
    Example usage : POST /historical_prices?ticker=AAPL  
    """
    ticker = ticker.upper()
    try : 
        data=read_db_v2(ticker=ticker, start_date=start_date, end_date=end_date, timeframe=timeframe)

        chartData =  []
        for row in data.index:
            data_row = data.loc[row]  
            chartData.append({
        "timeframe": data_row["timeframe"],
        "date": str(data_row["date"]),
        "close": float(data_row["close"])
        })
        
        return chartData

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

@app.post("/get_ohlc")
def get_ohlc_endpoint(
    ticker: str = Query(..., description="Ticker symbol"),
    start_date : str = Query(default=None, description="Start date interval (Optional)"),
    end_date : str= Query(default=None, description="End date interval(Optional)"),
    timeframe: str = Query(default="1d", description="Timeframe (Optional)"),
    ):
    """
    Get OHLC (Open, High, Low, Close) data for a ticker.
    
    Example: POST /get_ohlc?ticker=btc-usd&start_date=2024-01-01&end_date=2024-12-31&timeframe=1d
    """
   
   
    try:
        data = read_db_v2(ticker, start_date, end_date , timeframe)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
            
        ohlc_data = [
            {
                "date": str(data["date"][row]),
                "open": float(data["open"][row]),
                "high": float(data["high"][row]),
                "low": float(data["low"][row]),
                "close": float(data["close"][row]),
                "timeframe": str(data["timeframe"][row]),
            }
            for row in data.index
        ]
        return ohlc_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/get_mutiple_patterns", response_model=SubsequenceResponse)
def get_patterns(
    ticker: str = Query(..., description="Ticker symbol"), 
    start_date: str = Query(...),
    end_date: str = Query(...),
    k: int = Query(3, description="Number of pattern to return"),
    metric: str = Query("l2", description="Distance metric: 'l1' or 'l2'"),
    wrap: bool = Query(True, description="Allow wrapping (circular search)"),
    timeframe: str = Query("1d", description="Timeframe for the reference data ('1d', '1h', etc.)") ,
):
    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="Start date must be less than end date")

    try:
        
        query_data = read_db_v2(ticker, start_date, end_date, timeframe)
        
        if query_data.empty:
            raise HTTPException(status_code=404, detail="No data found for the given date range")

        reference_data = read_db_v2(ticker,timeframe)
        
        if reference_data.empty:
            raise HTTPException(status_code=404, detail="No data found for the given date range")
        
        
        query_data=query_data.sort_values("date")
        reference_data=reference_data.sort_values("date")
        
        query = query_data["close"].values
        array2 = reference_data["close"].values
        dates = reference_data["date"].values

        query_return = calculate_query_return(ticker, start_date, end_date)

        best_indices, best_dates, best_subarrays, best_distances, query, array2 = array_with_shift(
            query, array2, dates, k=k, metric=metric, wrap=wrap
        )

        matches = [
            SubsequenceMatch(
                dates=[str(d) for d in dates_],
                closes=[float(v) for v in values],
                similarity=to_float(dist),
                query_return=to_float(query_return),
                description= "btc on date"
            )
            for dates_, values, dist in zip(best_dates, best_subarrays, best_distances)
        ]

        return SubsequenceResponse(matches=matches)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern search failed: {str(e)}")


@app.post("/get_multiple_patterns_ohcl")
def get_patterns(
    ticker: str = Query(..., description="Ticker symbol"), 
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    k: int = Query(3, description="Number of patterns to return"),
    metric: str = Query("l2", description="Distance metric: 'l1' or 'l2'"),
    wrap: bool = Query(True, description="Allow wrapping (circular search)"),
    timeframe: str = Query("1d", description="Timeframe for the reference data ('1d', '1h', etc.)"),
):
    
    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="Start date must be less than end date")

    try:
       
        query_data = read_db_v2(ticker, start_date, end_date, timeframe)
        if query_data.empty:
            raise HTTPException(status_code=404, detail="No data found for the given date range")

       
        reference_data = read_db_v2(ticker, timeframe)
        if reference_data.empty:
            raise HTTPException(status_code=404, detail="No reference data found for the given timeframe")

        
        query_data = query_data.sort_values("date")
        reference_data = reference_data.sort_values("date")

        query = query_data["close"].values
        array2 = reference_data["close"].values
        dates = reference_data["date"].values

        
        query_return = calculate_query_return(ticker, start_date, end_date)

        
        best_indices, best_dates, best_subarrays, best_distances, query, array2 = array_with_shift(
            query, array2, dates, k=k, metric=metric, wrap=wrap
        )

        matches = []
        for idx, (indices, dates_, values, dist) in enumerate(zip(best_indices, best_dates, best_subarrays, best_distances)):
            
            if not isinstance(indices, (list, tuple, pd.Series)):
                indices = [indices]

            
            ohlc_segment = reference_data.iloc[indices][["date", "open", "high", "low", "close"]].copy()
            ohlc_segment["date"] = ohlc_segment["date"].astype(str)

            match = {
                "pattern_id": idx + 1,
                "dates": ohlc_segment["date"].tolist(),
                "opens": ohlc_segment["open"].astype(float).tolist(),
                "highs": ohlc_segment["high"].astype(float).tolist(),
                "lows": ohlc_segment["low"].astype(float).tolist(),
                "closes": ohlc_segment["close"].astype(float).tolist(),
                "similarity": float(dist),
                "query_return": float(query_return),
                "description": f"{ticker} pattern match {idx+1}"
            }

            matches.append(match)

        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "timeframe": timeframe,
            "query_return": float(query_return),
            "patterns": matches
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern search failed: {str(e)}")

