# Cascadian Airbnb

## Overview and motivation

As a frequent Airbnb guest in Vancouver BC, Seattle WA, and Portland OR, I wanted to learn more about Airbnb listings in these cities. More specifically, I wanted to know more about the price of listings and better understand guest satisfaction.


## Requirements

Python version is `3.7.6`. Packages and dependencies were managed using a conda virtual environment, these are found in `cascbnb.yaml` (see below)

## Files

- `getdata.sh`: Bash script for downloading and unzipping original datasets
for wrangling. In the terminal from the root directory

    sh getdata.sh

Use this if you want to just get the original datasets.

- `wrangle.py`: Python script for wrangling data and producing cleaned datasets. In the terminal, from the root directory
 
    python wrangle.py

- `wrangling.ipynb`: In this notebook we clean and prepare data for analysis and modeling. Produces same datasets as `wrangle.py`
- `analysis-and-modeling.ipynb`: In this notebook we explore answers to the questions above and summarize our findings.

- `cascbnb.yml`: YAML file for setting up conda virtual environment. You can create the environment with 

    conda env create -f cascbnb.yml


## Summary of results

### How do listing prices relate to location?

We found that while prices were distributed similarly across all three cities, Portland has the lowest median listing price while Seattle has the highest. As of November 2019, the median prices of listings (in USD) were Portland $90, Vancouver $97, Seattle $115.

We found some evidence that neighbourhoods closer to downtown were more expensive in all three cities. Among neighbourhoods with at least 10 listings, the 5 most and least expensive neighbourhoods to visit (by median price) in each city were

| Portland   							  |  Seattle  										 | Vancouver   					|
|---------------------------|--------------------------------|----------------------|
|Pearl  								$180|Central Business District $252.5|Downtown					$133|
|Downtown               $180|Pike-Market								 $243|Downtown-Eastside $124|
|Bridgeton              $139|International District			 $195|Kitsilano					$109|
|Grant Park       			$125|Briarcliff									 $190|West End       		$105|
|Goose Hollow           $120|First Hill 								 $169|Mount Pleasant		$102|


| Portland   							 |  Seattle  					        | Vancouver   					|
|--------------------------|----------------------------|-----------------------|
|Centennial  			 			$44|Meadowbrook							 $50|Oakridge						 $57|
|Mill Park              $45|Holly Park  	      	   $55|Renfrew-Collingwood $60|
|Lents                  $50|Dunlap   	        			 $59|Marpole						 $68|
|Madison South         	$50|Pinehurst								 $60|Victoria-Fraserview $68|
|Hazelwood              $55|North College Park       $65|Killarney					 $72|


### How do listing prices change over time?

We found expected weekly and seasonal trends in overall price, with lows in November, a short-lived spike around the winter holidays and a steady increase in price peaking in late summer. These trends were similar for all three cities, although the summer increase in price was more pronounced for Seattle. Portland has an unusual large spike in price on October 10, 2020. 

### How does overall guest satisfaction relate to location?

Ratings overall were very high, with 50% of listings recieving a rating of 97 or above. We found that Portland has higher listing ratings than Seattle or Vancouver. As of November 2019, the median ratings were 98 (Portland), 97 (Seattle), and 97 (Vancouver). 

	Among neighbourhoods with at least 10 listings, the 5 neighbourhoods with highest and lowest ratings in each city were

| Portland   							 |  Seattle  					| Vancouver   					|
|--------------------------|--------------------|-----------------------|
|West Portland Park  		100|Interbay    	   100|Mount Pleasant				98|
|Alameda                 99|Laurelhurst  	  99.5|Riley Park					 	98|
|Concordia               99|Windermere   	  99.5|Arbutus Ridge				97|
|Hosford-Abernethy       99|East Queen Anne   99|Downtown Eastside		97|
|Humboldt                99|Fairmount Park    99|Dunbar Southlands		97|


| Portland   							 |  Seattle  					        | Vancouver   					|
|--------------------------|----------------------------|-----------------------|
|Hayhurst  			 				 95|International District  92.5|Killarney						94|
|Centennial              95|Holly Park  	      	    93|Marpole					 	  95|
|Bridgeton             95.5|Pioneer Square   	        94|Oakridge							95|
|Mill Park         			 96|University District     94.5|Shaugnessy						95|
|Portland Downtown       96|Belltown      					  95|Victoria-Fraserview	95|

### How does overall guest satisfaction relate to location?

Interestingly, we failed to find a meaningful association between neighbourhood median price and median ratings

![nb_med_price_vs_ratings]({{site.baseurl}}/assets/img/blog/nb_med_price_vs_ratings.png)

This is pretty good news for the budget traveler!

### Which listing features are most closely related to guest satisfaction?

Among the most important listing features related to guest satisfaction were

- The number of reviews a listing has, as well as the the number of reviews per month.
- Price, cleaning fee and security deposit.
- How long the host has been hosting and their response rate.
- The number of amenities the listing has.
- How many people the listing accommodates and how many extra people are allowed.


### Acknowledgments

Special thanks to [InsideAirvnb](http://insideairbnb.com/get-the-data.html) for providing the data, and to the many  contributors and maintainers of the open-source tools used for this project.
