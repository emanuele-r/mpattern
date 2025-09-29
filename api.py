from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from api_function import process_data, optimize_calc, array_with_shift, dynamic_time_warping, get_data
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, Response
from datetime import datetime, time
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


class HistoricalPrice(BaseModel):
    date: str
    close: float


class Match(BaseModel):
    dates: List[str]
    values: List[float]
    distance: float


class PricesResponse(BaseModel):
    matches: List[Match]


class HistoricalPricesResponse(BaseModel):
    prices: List[HistoricalPrice]


class OptimizeResponse(BaseModel):
    indices: List[int]
    dates: List[str]
    values: List[float]


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

@app.post("/historical_prices", response_model=HistoricalPricesResponse)
def read_data(
    ticker: str = Query(..., description="Ticker symbol")):
    """
    Example usage : GET /historical_prices
    """
    ticker.upper()
    try:
        if  os.path.exists(f"{ticker}1D.csv"):
            data = process_data(ticker)
            prices = [
                HistoricalPrice(date=str(data["Date"][i]), close=float(data["Close"][i]))
                for i in data.index
            ]
            return HistoricalPricesResponse(prices=prices)
        else :
            downloaded_data = get_data(ticker, start_date="2008-01-01", end_date=datetime.datetime.now().strftime("%Y-%m-%d"), interval="1d")
            time.sleep(10)
            series =process_data(ticker)
            prices = [
                HistoricalPrice(date=str(series["Date"][i]), close=float(series["Close"][i]))
                for i in data.index
            ]
            return HistoricalPricesResponse(prices=prices)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    





@app.post("/update_price", response_model=OptimizeResponse)
def update_date(start_date: str = Query(...), end_date: str = Query(...)):
    """
    Example usage : POST /update_price?start_date=2025-08-01&end_date=2025-08-31
    """
    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="Start date must be less than end date")
    try:
        best_indices, best_dates, best_subarray, query, array2, time_series = optimize_calc(start_date, end_date)

        return OptimizeResponse(
            indices=[int(idx) for idx in best_indices],
            dates=[str(d) for d in best_dates],
            values=[float(v) for v in best_subarray]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@app.post("/get_pattern", response_model=PricesResponse)
def get_patterns(
    start_date: str = Query(...),
    end_date: str = Query(...),
    k: int = Query(3, description="Number of top motifs to return"),
    metric: str = Query("l2", description="Distance metric: 'l1' or 'l2'"),
    wrap: bool = Query(True, description="Allow wrapping (circular search)"),
    ticker: str = Query(..., description="Ticker symbol") , 
):
    """
    Example usage:
    POST /get_pattern?start_date=2025-08-01&end_date=2025-08-31&k=5&metric=l1
    """
    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="Start date must be less than end date")

    try:
        data = process_data()
        query = data.loc[(data["Date"] >= start_date) & (data["Date"] <= end_date), "Close"].values
        array2 = data["Close"].values
        dates = data["Date"].values

        best_indices, best_dates, best_subarrays, best_distances, query, array2 = array_with_shift(query, array2, dates, k=k, metric=metric, wrap=wrap)


        matches = [
            Match(
                dates=[str(d) for d in dates_],
                values=[float(v) for v in values],
                distance=float(dist)
            )
            for dates_, values, dist in zip(best_dates, best_subarrays, best_distances)
        ]

        return PricesResponse(matches=matches)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_dynamic_time_pattern", response_model=PricesResponse)
def get_dynamic_pattern(
    start_date: str = Query(...),
    end_date: str = Query(...),
    k: int = Query(3, description="Number of top motifs to return"),
    metric: str = Query("l2", description="Distance metric: 'l1' or 'l2'"),
    wrap: bool = Query(True, description="Allow wrapping (circular search)"), 
    length_tolerance: int = Query(0, description="Maximum length difference between query and reference arrays"),
    ticker: str = Query(..., description="Ticker symbol")
):  
    """
    Example usage:
    POST /get_dynamic_time_pattern?start_date=2025-08-01&end_date=2025-08-31&k=5&metric=l1&ticker=EURUSD
    """
    
    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="Start date must be less than end date")

    try:
        data=process_data(ticker)
        query = data.loc[(data["Date"] >= start_date) & (data["Date"] <= end_date), "Close"].values
        array2 = data["Close"].values
        dates = data["Date"].values

        best_indices, best_dates, best_subarrays, best_distances, query, array2 = dynamic_time_warping(query, array2, dates, k=k, metric=metric, wrap=wrap, length_tolerance=length_tolerance)

        matches = [
            Match(
                dates=[str(d) for d in dates_],
                values=[float(v) for v in values],
                distance=float(dist)
            )
            for dates_, values, dist in zip(best_dates, best_subarrays, best_distances)
        ]
        return PricesResponse(matches=matches)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))