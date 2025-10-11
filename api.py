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


@app.get("/get_ticker_list")
def get_tickers():
   
    try:
        data = read_ticker_list()  
        categoryTypes = []
        for row in data.index:
            category= data["category"][row]
            ticker =data["ticker"][row]
            categoryTypes.append({"category": category, "ticker": ticker})
        
        tickers = []
        for row in data.index :
            category = data["category"][row]
            symbol = data["ticker"][row]
            price = data["close"][row]
            change = data["change"][row]
            tickers.append({"category": category, "symbol": symbol, "price": price, "change": change})
        
        prices = {"categoryTypes" : categoryTypes ,  "tickers" : tickers}
       
        
        return prices
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/historical_prices")
def read_data(
    ticker: str = Query(..., description="Ticker symbol"), 
    start_date : str = Query(default=None, description="Start date interval (Optional)"),
    end_date : str= Query(default=None, description="End date interval(Optional)"),
    timeframe :str =Query(default="1d", description="Timeframe (Optional)")
    ):
    """
    Example usage : POST /historical_prices?ticker=AAPL  
    """
    ticker = ticker.upper()
    try : 
        data=read_db_v2(ticker, start_date, end_date, timeframe)

        chartData =  []
        for row in data.index:
            data_row = data.loc[row]  
            chartData.append({
        "timeframe": timeframe,
        "date": str(data_row["date"]),
        "close": float(data_row["close"])
        })
        
        return chartData

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/chartData")
def get_chartData(
    timeFrame : str = Query(..., description="Time frame"),
    symbol :str =Query(..., description="Ticker symbol"),
    ) :
    ticker = ticker.upper()
    try : 
        data=read_db_v2(symbol,  timeFrame)

        chartData =  []
        for row in data.index:
            data_row = data.loc[row]  
            chartData.append({
        "timeframe": timeFrame ,
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
    end_date : str= Query(default=None, description="End date interval(Optional)")):

    try:
        data = read_db_v2(ticker, start_date, end_date)
            
        datas = [
            {
                "date": str(data["date"][row]),
                "open": float(data["open"][row]),
                "high": float(data["high"][row]),
                "low": float(data["low"][row]),
                "close": float(data["close"][row]),
            }
            for row in data.index
        ]
        return datas
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    
@app.post("/get_single_pattern", response_model=SubsequenceResponse)
def get_single_pattern(
    ticker: str = Query(..., description="Ticker symbol"),
    start_date: str = Query(...),
    end_date: str = Query(...),
):
    ticker= ticker.upper()
    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="Start date must be less than end date")

    try:
        read_db(ticker, start_date, end_date)
            
        best_indices, best_dates, best_subarray, query, array2, time_series, best_distance = optimize_calc(
            ticker, start_date, end_date
        )
        
        query_return = calculate_query_return(ticker, start_date, end_date)

        matches = []
        match = SubsequenceMatch(
            dates=[str(d) for d in best_dates],
            closes=[float(v) for v in best_subarray],
            similarity=to_float(best_distance),
            query_return=to_float(query_return),
            description="" 
        )
        matches.append(match)
               
        return SubsequenceResponse(matches=matches)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update price failed: {str(e)}")



@app.post("/get_mutiple_patterns", response_model=SubsequenceResponse)
def get_patterns(
    ticker: str = Query(..., description="Ticker symbol"), 
    start_date: str = Query(...),
    end_date: str = Query(...),
    k: int = Query(3, description="Number of top motifs to return"),
    metric: str = Query("l2", description="Distance metric: 'l1' or 'l2'"),
    wrap: bool = Query(True, description="Allow wrapping (circular search)"),
):
    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="Start date must be less than end date")

    try:
        
        data = read_db(ticker)
        
        query = data.loc[(data["date"] >= start_date) & (data["date"] <= end_date), "close"].values
        array2 = data["close"].values
        dates = data["date"].values

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




@app.post("/get_dynamic_time_pattern", response_model=SubsequenceResponse)
def get_dynamic_pattern(
    ticker: str = Query(..., description="Ticker symbol"),
    start_date: str = Query(...),
    end_date: str = Query(...),
    k: int = Query(3, description="Number of top motifs to return"),
    metric: str = Query("l2", description="Distance metric: 'l1' or 'l2'"),
    wrap: bool = Query(True, description="Allow wrapping (circular search)"), 
    length_tolerance: int = Query(0, description="Maximum length difference between query and reference arrays"),
):  
    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="Start date must be less than end date")

    try:
        if os.path.exists(f"{ticker}1D.csv"):
            data = process_data(ticker)
        else:
            get_data(ticker, start_date="2008-01-01", end_date=datetime.now().strftime("%Y-%m-%d"), interval="1d")
            data = process_data(ticker)

        query = data.loc[(data["Date"] >= start_date) & (data["Date"] <= end_date), "Close"].values
        array2 = data["Close"].values
        dates = data["Date"].values

        best_indices, best_dates, best_subarrays, best_distances, query, array2 = dynamic_time_warping(
            query, array2, dates, k=k, metric=metric, wrap=wrap, length_tolerance=length_tolerance
        )
        
        query_return = calculate_query_return(ticker, start_date, end_date)

        matches = [
            SubsequenceMatch(
                dates=[str(d) for d in dates_],
                closes=[float(v) for v in values],
                similarity=to_float(dist),
                query_return=[to_float(query_return) for d in query_return],
                description=""
            )
            for dates_, values, dist in zip(best_dates, best_subarrays, best_distances)
        ]

        return SubsequenceResponse(matches=matches)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DTW failed: {str(e)}")
