curl -s https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt | \
awk -F'|' 'NR>1 && $1 !~ /^File Creation Time/ {print $1}' | \
while read ticker; do 
    echo "insert or replace into ticker_list (ticker, category, change, close) values ('$ticker', 'stock', '0' , '0');"
done | sqlite3 asset_prices.db
