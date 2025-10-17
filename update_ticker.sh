curl -s https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt | \
awk -F'|' 'NR>1 && $1 !~ /^File Creation Time/ {print $1}' | \
head -20 | \
while read ticker; do 
    echo "insert or replace into symbols (ticker) values ('$ticker');"
done | sqlite3 asset_prices.db
