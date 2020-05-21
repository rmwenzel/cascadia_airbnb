## Overview and motivation

As a frequent Airbnb guest in Vancouver BC, Seattle WA, and Portland OR, I wanted to learn more about Airbnb listings in these cities. More specifically, I wanted to know more about the price of listings and better understand guest satisfaction.

I used the most recent AirBnb data available for all three cities (February 2020) from [InsideAirBnb](http://insideairbnb.com/get-the-data.html) and cleaned and analyzed it using standard Python tools.

For easier to read versions of the notebooks, see the [project website](https://rmwenzel.com/cascadia-airbnb/)

## Requirements

Python version is `3.7.6`. Packages and dependencies were managed using a conda virtual environment, these are found in `cascbnb.yaml` (see below)

## Files

- `getdata.sh`: Bash script for downloading and unzipping original datasets
for wrangling. In the terminal, from the root directory

    ```sh getdata.sh```

Use this if you want to just get the original datasets.

- `wrangle.py`: Python script for wrangling data and producing cleaned datasets. In the terminal, from the root directory
 
  ```python wrangle.py```

- `wrangling.ipynb`: In this notebook we clean and prepare data for analysis and modeling. Produces same datasets as `wrangle.py`
- `analysis-and-modeling.ipynb`: In this notebook we explore answers to the questions above and summarize our findings.

- `cascbnb.yml`: YAML file for setting up conda virtual environment. You can create the environment with 

    ```conda env create -f cascbnb.yml```


## Summary of results

### How do listing prices relate to location?

We found that while prices were distributed similarly across all three cities, Portland has the lowest median listing price while Seattle has the highest. Portland had the lowest listing price while Seattle had the highest - the median prices of listings (in USD) were Portland $90, Vancouver $95, Seattle $115.

We found some evidence that neighbourhoods closer to downtown were more expensive in all three cities. Among neighbourhoods with at least 10 listings, the 5 most and least expensive neighbourhoods (by median price) in each city were

|Portland   							   |Seattle  								  		   | Vancouver   					 |
|---------------------------:|--------------------------------:|----------------------:|
|Pearl  								 $180|Central Business District  $285.5|Downtown				 $120.5|
|Downtown                $169|Pike-Market								   $201|Downtown-Eastside  $113|
|Northwest District      $120|International District			 $195|Kitsilano					 $112|
|Goose Hollow      		 $119.5|Pioneer Square						 $187.5|Mount Pleasant   $103.5|
|Bridgeton               $120|Briarcliff 								   $182|West End		       $100|


| Portland   							  |  Seattle  					        | Vancouver   					 |
|--------------------------:|----------------------------:|-----------------------:|
|Mill Park  			 		 	 $44|Meadowbrook							 $51|Renfrew-Collingwood	$57|
|Centennial              $45|Dunlap  	      	         $60|Oakridge						  $60|
|Lents                   $50|High Point   	        	 $65|Victoria-Fraserview	$68|
|Madison South         	 $50|North College Park				 $65|Killarney   				  $68|
|Parkrose                $55|Pinehurst                 $65|Marpole	  					$72|


### How do listing prices change over time?

Since historical booking data was unavailable, to look at price trends we analyzed the booking calendar, that is the price to book each listing for each day for the next 12 months.

We found an expected weekly trend in overall price, as well as a seasonal trend, with a steady increase in price from early spring to summer. Price declined again reaching a low around the 2021 winter holidays but then strangely rising again in January-February 2021.

Price by city showed similar patterns -- weekly price fluctuations and a steady increase reaching a maximum in late summer but there were some clear differences. The weekly price trend is much less pronounced in Vancouver than Seattle and Portland, while the summer price trend is much more pronounced in Seattle than Vancouver or Portland. Seattle and Portland prices both show the January-February 2021 increase we saw overall but Vancouver price does not.

The January-February 2021 price increase is most pronounced in Portland in fact at its peak it exceeds peak summer price by around $10 (about 8%). During this time, Portland median price exceeds Vancouver's.

### How does overall guest satisfaction relate to location?

Ratings overall were very high, with 50% of listings recieving a rating of 97 or above. We found that Portland has higher listing ratings than Seattle or Vancouver. We found that Portland had higher listing ratings than Seattle or Vancouver - the median ratings were Portland 98, Seattle 97, and Vancouver 97. 

Among neighbourhoods with at least 10 listings, the 5 neighbourhoods with highest and lowest ratings in each city were

| Portland   							 |  Seattle  					| Vancouver   					|
|-------------------------:|-------------------:|----------------------:|
|West Portland Park  		100|Interbay    	   100|Mount Pleasant				98|
|Alameda                 99|Laurelhurst  	  99.5|Riley Park					 	98|
|Centennial              99|Crown Hill   	    99|Arbutus Ridge				97|
|Concordia     					 99|East Queen Anne   99|Downtown Eastside		97|
|Eastmoreland            99|Fairmount Park    99|Dunbar Southlands		97|

| Portland   							 |  Seattle  					        | Vancouver   					|
|-------------------------:|---------------------------:|----------------------:|
|Hayhurst  			 				 95|International District  91.5|Killarney					94.5|
|Bridgeton               96|Pioneer Square  	      	92|Marpole					 	  95|
|Mill Park               96|South Beacon Hill  	    93.5|Shaugnessy						95|
|Old Town/Chinatown      96|Central Business District 95|Victoria-Fraserview	95|
|Powellhurst-Gilbert     96|Pinehurst      					  95|Arbutus Ridge				96|

### How do listing prices in neighbourhoods relate to guest satisfaction?

We found weak evidence of a negative relationship between price and rating overall, and in Portland and Seattle, with Seattle showing the stronger relationship. We found strong evidence of a positive relationship between price and rating in Vancouver.


### Which listing features are most closely related to guest satisfaction?

Among the most important listing features related to guest satisfaction were

- The number of reviews a listing has, as well as the the number of reviews per month.
- Price, cleaning fee and security deposit.
- How long the host has been hosting and their response and acceptance rate.
- The number of amenities the listing has.
- Whether the listing has a coffee maker!


### Acknowledgments

Special thanks to [InsideAirbnb](http://insideairbnb.com/get-the-data.html) for providing the data, and to the many  contributors and maintainers of the open-source tools used for this project.
