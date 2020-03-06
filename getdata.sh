#!/bin/bash

mkdir ./data/
cd ./data/

## portland data
wget http://data.insideairbnb.com/united-states/or/portland/2020-02-13/data/listings.csv.gz
gunzip listings.csv.gz
mv listings.csv portland_listings.csv
wget http://data.insideairbnb.com/united-states/or/portland/2020-02-13/data/calendar.csv.gz
gunzip calendar.csv.gz
mv calendar.csv portland_calendar.csv
wget http://data.insideairbnb.com/united-states/or/portland/2020-02-13/visualisations/neighbourhoods.geojson
mv neighbourhoods.geojson portland_neighbourhoods.geojson

## seattle data
wget http://data.insideairbnb.com/united-states/wa/seattle/2020-02-22/data/listings.csv.gz
gunzip listings.csv.gz
mv listings.csv seattle_listings.csv
wget http://data.insideairbnb.com/united-states/wa/seattle/2020-02-22/data/calendar.csv.gz
gunzip calendar.csv.gz
mv calendar.csv seattle_calendar.csv
wget http://data.insideairbnb.com/united-states/wa/seattle/2020-02-22/visualisations/neighbourhoods.geojson
mv neighbourhoods.geojson seattle_neighbourhoods.geojson

## vancouver data
wget http://data.insideairbnb.com/canada/bc/vancouver/2020-02-16/data/listings.csv.gz
gunzip listings.csv.gz
mv listings.csv vancouver_listings.csv
wget http://data.insideairbnb.com/canada/bc/vancouver/2020-02-16/data/calendar.csv.gz
gunzip calendar.csv.gz
mv calendar.csv vancouver_calendar.csv
wget http://data.insideairbnb.com/canada/bc/vancouver/2020-02-16/visualisations/neighbourhoods.geojson
mv neighbourhoods.geojson vancouver_neighbourhoods.geojson
