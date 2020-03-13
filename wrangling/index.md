---
layout: page
title: Data wrangling
---

In this notebook we clean and tidy our data, and save analysis and modeling for [another notebook](./analysis-and-modeling.ipynb)

## Contents

- [Gather and assess](#gather-and-assess)
- [Clean and tidy](#clean-and-tidy)
  - [`listings` dataset](#listings-dataset)
    - [Drop `listings` rows or columns](#drop-listings-rows-or-columns)
    - [Enforce `listings`dtypes](#enforce-listings-dtypes)
    - [Missing `listings` values](#missing-listings-values)
    - [Downcast `listings` dtypes](#downcast-listings-dtypes)
  - [`calendar` dataset](#calendar-dataset)
    - [Drop `calendar` rows or columns](#drop-calendar-rows-or-columns)
    - [Enforce `calendar` dtypes](#enforce-calendar-dtypes)
    - [Missing `calendar` values](#missing-calendar-values)
    - [Synchronize `calendar.date`](#synchronize-calendar-date)
  - [Create amenities features](#create-amenities-features)
  - [Currency conversion](#currency-conversion)
  - [Save cleaned datasets](#save-cleaned-datasets)


```python
# standard imports
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style('darkgrid')
import re
import os

from collections import defaultdict
from pandas.api.types import CategoricalDtype
from functools import partial
from matplotlib.axes._axes import _log as matplotlib_axes_logger

# settings
pd.set_option('display.max_columns', None)
matplotlib_axes_logger.setLevel('ERROR')
```

## Gather and assess 

The data we're using was compiled by [InsideAirbnb](http://insideairbnb.com/) from information made publicly available on [Airbnb](http://airbnb.com). Some interesting disclaimers about the data can be found at the [about page](http://insideairbnb.com/about.html) as well as more about Inside Airbnb's model for estimating occupancy rate. You can find datasets for a very large number of major cities worldwide [here](http://insideairbnb.com/about.html).

For each city, InsideAirbnb provides data sets which are updated roughly monthly. Let's collect the most recent available for Vancouver (February 16, 2020), Seattle (February 22, 2020), and Portland (February 13, 2020).


```python
# download and unzip datasets for each city
os.system('sh getdata.sh');
```

After unzipping and renaming, we have 3 data files for each city.


|File Name               | Description                                |
|------------------------|--------------------------------------------|
| listings.csv           | Detailed Listings data                     |
| calendar.csv           | Detailed Calendar data for listings        |
| neighbourhoods.geojson | GeoJSON file of neighbourhoods of the city.|

These can all be found in the `/data` directory.


```python
def load_dfs(file_prefixes, city_names, files_path):
    """Load all but .geojson data files for each city into nested dict of dataframes."""
    dfs = defaultdict(dict)

    for file_prefix in file_prefixes:
        for city_name in city_names:
            file_path = files_path + city_name + '_' + file_prefix + '.csv'
            dfs[file_prefix][city_name] = pd.read_csv(file_path)

    return dfs

file_prefixes = ['listings', 'calendar']
city_names = ['seattle', 'portland', 'vancouver']
files_path = 'data/'
dfs = load_dfs(file_prefixes, city_names, files_path)
```

Some of the questions we're interested in involve comparisons between the three cities, so we'll merge the corresponding dataframes for each city into a single large dataframe.


```python
def merge_dfs(dfs, file_prefixes, city_names):
    """Merge dataframes for each kind of data file."""
    merged_dfs = defaultdict(list)
    
    for file_prefix in file_prefixes:
        dfs_list = [dfs[file_prefix][city_name] for city_name in city_names]
        merged_dfs[file_prefix] = pd.concat(dfs_list, keys=city_names, names=['city'])
        try:
            merged_dfs[file_prefix].drop(columns=['city'], inplace=True)
        except:
            pass

    return merged_dfs
        
dfs = merge_dfs(dfs, file_prefixes, city_names)
```

The main dataset we'll be relying on is the `listings` dataset.


```python
listings_df = dfs['listings']
listings_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>id</th>
      <th>listing_url</th>
      <th>scrape_id</th>
      <th>last_scraped</th>
      <th>name</th>
      <th>summary</th>
      <th>space</th>
      <th>description</th>
      <th>experiences_offered</th>
      <th>neighborhood_overview</th>
      <th>notes</th>
      <th>transit</th>
      <th>access</th>
      <th>interaction</th>
      <th>house_rules</th>
      <th>thumbnail_url</th>
      <th>medium_url</th>
      <th>picture_url</th>
      <th>xl_picture_url</th>
      <th>host_id</th>
      <th>host_url</th>
      <th>host_name</th>
      <th>host_since</th>
      <th>host_location</th>
      <th>host_about</th>
      <th>host_response_time</th>
      <th>host_response_rate</th>
      <th>host_acceptance_rate</th>
      <th>host_is_superhost</th>
      <th>host_thumbnail_url</th>
      <th>host_picture_url</th>
      <th>host_neighbourhood</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>host_verifications</th>
      <th>host_has_profile_pic</th>
      <th>host_identity_verified</th>
      <th>street</th>
      <th>neighbourhood</th>
      <th>neighbourhood_cleansed</th>
      <th>neighbourhood_group_cleansed</th>
      <th>state</th>
      <th>zipcode</th>
      <th>market</th>
      <th>smart_location</th>
      <th>country_code</th>
      <th>country</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>is_location_exact</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>bed_type</th>
      <th>amenities</th>
      <th>square_feet</th>
      <th>price</th>
      <th>weekly_price</th>
      <th>monthly_price</th>
      <th>security_deposit</th>
      <th>cleaning_fee</th>
      <th>guests_included</th>
      <th>extra_people</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>minimum_minimum_nights</th>
      <th>maximum_minimum_nights</th>
      <th>minimum_maximum_nights</th>
      <th>maximum_maximum_nights</th>
      <th>minimum_nights_avg_ntm</th>
      <th>maximum_nights_avg_ntm</th>
      <th>calendar_updated</th>
      <th>has_availability</th>
      <th>availability_30</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>availability_365</th>
      <th>calendar_last_scraped</th>
      <th>number_of_reviews</th>
      <th>number_of_reviews_ltm</th>
      <th>first_review</th>
      <th>last_review</th>
      <th>review_scores_rating</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>requires_license</th>
      <th>license</th>
      <th>jurisdiction_names</th>
      <th>instant_bookable</th>
      <th>is_business_travel_ready</th>
      <th>cancellation_policy</th>
      <th>require_guest_profile_picture</th>
      <th>require_guest_phone_verification</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>reviews_per_month</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">seattle</th>
      <th>0</th>
      <td>2318</td>
      <td>https://www.airbnb.com/rooms/2318</td>
      <td>20200222045957</td>
      <td>2020-02-22</td>
      <td>Casa Madrona - Urban Oasis 1 block from the park!</td>
      <td>Gorgeous, architect remodeled, Dutch Colonial ...</td>
      <td>This beautiful, gracious home has been complet...</td>
      <td>Gorgeous, architect remodeled, Dutch Colonial ...</td>
      <td>none</td>
      <td>Madrona is a hidden gem of a neighborhood. It ...</td>
      <td>We adhere to a 10pm -9am quiet hour schedule, ...</td>
      <td>NaN</td>
      <td>Guests can access any part of the house.</td>
      <td>We are a family who live next door and are ava...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/im/pictures/02973ad3-a...</td>
      <td>NaN</td>
      <td>2536</td>
      <td>https://www.airbnb.com/users/show/2536</td>
      <td>Megan</td>
      <td>2008-08-26</td>
      <td>Seattle, Washington, United States</td>
      <td>I welcome guests from all walks of life and ev...</td>
      <td>within an hour</td>
      <td>100%</td>
      <td>92%</td>
      <td>t</td>
      <td>https://a0.muscache.com/im/pictures/user/016a1...</td>
      <td>https://a0.muscache.com/im/pictures/user/016a1...</td>
      <td>Minor</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>['email', 'phone', 'reviews', 'jumio', 'offlin...</td>
      <td>t</td>
      <td>f</td>
      <td>Seattle, WA, United States</td>
      <td>Madrona</td>
      <td>Madrona</td>
      <td>Central Area</td>
      <td>WA</td>
      <td>98122</td>
      <td>Seattle</td>
      <td>Seattle, WA</td>
      <td>US</td>
      <td>United States</td>
      <td>47.61082</td>
      <td>-122.29082</td>
      <td>t</td>
      <td>House</td>
      <td>Entire home/apt</td>
      <td>9</td>
      <td>2.5</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>Real Bed</td>
      <td>{Internet,Wifi,Kitchen,"Free parking on premis...</td>
      <td>NaN</td>
      <td>$296.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>$500.00</td>
      <td>$250.00</td>
      <td>8</td>
      <td>$25.00</td>
      <td>7</td>
      <td>1000</td>
      <td>7</td>
      <td>7</td>
      <td>1000</td>
      <td>1000</td>
      <td>7.0</td>
      <td>1000.0</td>
      <td>4 weeks ago</td>
      <td>t</td>
      <td>6</td>
      <td>36</td>
      <td>65</td>
      <td>65</td>
      <td>2020-02-22</td>
      <td>32</td>
      <td>11</td>
      <td>2008-09-15</td>
      <td>2020-02-01</td>
      <td>100.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>t</td>
      <td>STR-OPLI-19-002837</td>
      <td>{WASHINGTON," Seattle"," WA"}</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>f</td>
      <td>f</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6606</td>
      <td>https://www.airbnb.com/rooms/6606</td>
      <td>20200222045957</td>
      <td>2020-02-22</td>
      <td>Fab, private seattle urban cottage!</td>
      <td>This tiny cottage is only 15x10, but it has ev...</td>
      <td>Soo centrally located, this is a little house ...</td>
      <td>This tiny cottage is only 15x10, but it has ev...</td>
      <td>none</td>
      <td>A peaceful yet highly accessible neighborhood,...</td>
      <td>Check in is at three, if you'd like a snack or...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>I often escape to kingston and my place on the...</td>
      <td>Please treat the cottage as if it were your ow...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/im/pictures/45742/2111...</td>
      <td>NaN</td>
      <td>14942</td>
      <td>https://www.airbnb.com/users/show/14942</td>
      <td>Joyce</td>
      <td>2009-04-26</td>
      <td>Seattle, Washington, United States</td>
      <td>I am a therapist/innkeeper.I know my city well...</td>
      <td>within a few hours</td>
      <td>90%</td>
      <td>95%</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/users/14942/profile...</td>
      <td>https://a0.muscache.com/im/users/14942/profile...</td>
      <td>Wallingford</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>['email', 'phone', 'facebook', 'reviews', 'kba']</td>
      <td>t</td>
      <td>t</td>
      <td>Seattle, WA, United States</td>
      <td>Wallingford</td>
      <td>Wallingford</td>
      <td>Other neighborhoods</td>
      <td>WA</td>
      <td>98103</td>
      <td>Seattle</td>
      <td>Seattle, WA</td>
      <td>US</td>
      <td>United States</td>
      <td>47.65411</td>
      <td>-122.33761</td>
      <td>t</td>
      <td>Guesthouse</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>{TV,Internet,Wifi,"Air conditioning",Kitchen,"...</td>
      <td>NaN</td>
      <td>$90.00</td>
      <td>$670.00</td>
      <td>NaN</td>
      <td>$200.00</td>
      <td>$40.00</td>
      <td>2</td>
      <td>$10.00</td>
      <td>30</td>
      <td>1125</td>
      <td>30</td>
      <td>30</td>
      <td>1125</td>
      <td>1125</td>
      <td>30.0</td>
      <td>1125.0</td>
      <td>3 months ago</td>
      <td>t</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>20</td>
      <td>2020-02-22</td>
      <td>150</td>
      <td>16</td>
      <td>2009-07-17</td>
      <td>2019-09-28</td>
      <td>92.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>t</td>
      <td>NaN</td>
      <td>{WASHINGTON," Seattle"," WA"}</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>f</td>
      <td>f</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9419</td>
      <td>https://www.airbnb.com/rooms/9419</td>
      <td>20200222045957</td>
      <td>2020-02-22</td>
      <td>Glorious sun room w/ memory foambed</td>
      <td>This beautiful double room features a magical ...</td>
      <td>Our new Sunny space has a private room from th...</td>
      <td>This beautiful double room features a magical ...</td>
      <td>none</td>
      <td>Lots of restaurants (see our guide book) bars,...</td>
      <td>This area is an arts district,you will see all...</td>
      <td>Car 2 go is in this neigborhood Bus is across ...</td>
      <td>24 /7 access kitchen, bathroom and community s...</td>
      <td>I have a hands on warm approach to guests but ...</td>
      <td>No drugs,no smoking inside *outside in front o...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/im/pictures/56645186/e...</td>
      <td>NaN</td>
      <td>30559</td>
      <td>https://www.airbnb.com/users/show/30559</td>
      <td>Angielena</td>
      <td>2009-08-09</td>
      <td>Seattle, Washington, United States</td>
      <td>I am a visual artist who is  the  director of ...</td>
      <td>within a few hours</td>
      <td>100%</td>
      <td>92%</td>
      <td>t</td>
      <td>https://a0.muscache.com/im/users/30559/profile...</td>
      <td>https://a0.muscache.com/im/users/30559/profile...</td>
      <td>Georgetown</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>['email', 'phone', 'reviews', 'jumio', 'offlin...</td>
      <td>t</td>
      <td>t</td>
      <td>Seattle, WA, United States</td>
      <td>Georgetown</td>
      <td>Georgetown</td>
      <td>Other neighborhoods</td>
      <td>WA</td>
      <td>98108</td>
      <td>Seattle</td>
      <td>Seattle, WA</td>
      <td>US</td>
      <td>United States</td>
      <td>47.55017</td>
      <td>-122.31937</td>
      <td>t</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>{Internet,Wifi,"Air conditioning",Kitchen,"Fre...</td>
      <td>200.0</td>
      <td>$62.00</td>
      <td>$580.00</td>
      <td>$1,500.00</td>
      <td>$100.00</td>
      <td>$20.00</td>
      <td>1</td>
      <td>$15.00</td>
      <td>2</td>
      <td>180</td>
      <td>2</td>
      <td>2</td>
      <td>180</td>
      <td>180</td>
      <td>2.0</td>
      <td>180.0</td>
      <td>4 weeks ago</td>
      <td>t</td>
      <td>13</td>
      <td>43</td>
      <td>73</td>
      <td>348</td>
      <td>2020-02-22</td>
      <td>148</td>
      <td>17</td>
      <td>2010-07-30</td>
      <td>2019-12-27</td>
      <td>93.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>t</td>
      <td>str-opli-19-003039</td>
      <td>{WASHINGTON," Seattle"," WA"}</td>
      <td>f</td>
      <td>f</td>
      <td>moderate</td>
      <td>t</td>
      <td>t</td>
      <td>7</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>1.27</td>
    </tr>
  </tbody>
</table>
</div>




```python
listings_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 18429 entries, ('seattle', 0) to ('vancouver', 6205)
    Columns: 105 entries, id to reviews_per_month
    dtypes: float64(21), int64(21), object(63)
    memory usage: 14.9+ MB


There are 19997 total listings, with 105 columns. Columns are well named and descriptive enough to make sense of the variables they correspond to. We note that lot of these variables are long text strings. This dataset provides a lot of interesting opportunities for natural language processing.

We'll also be making use of the `calendar` dataset


```python
calendar_df = dfs['calendar']
calendar_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
      <th>adjusted_price</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">seattle</th>
      <th>0</th>
      <td>788146</td>
      <td>2020-02-22</td>
      <td>f</td>
      <td>$68.00</td>
      <td>$68.00</td>
      <td>30.0</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>340706</td>
      <td>2020-02-22</td>
      <td>t</td>
      <td>$90.00</td>
      <td>$90.00</td>
      <td>2.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>340706</td>
      <td>2020-02-23</td>
      <td>t</td>
      <td>$90.00</td>
      <td>$90.00</td>
      <td>2.0</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
calendar_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 6726585 entries, ('seattle', 0) to ('vancouver', 2265189)
    Data columns (total 7 columns):
     #   Column          Dtype  
    ---  ------          -----  
     0   listing_id      int64  
     1   date            object 
     2   available       object 
     3   price           object 
     4   adjusted_price  object 
     5   minimum_nights  float64
     6   maximum_nights  float64
    dtypes: float64(2), int64(1), object(4)
    memory usage: 412.3+ MB


There are 7298905 calendar entries, with 7 variables describing them. Again column names are descriptive.

## Clean and tidy


#### `listings` dataset

##### Drop `listings` rows or columns

First we check for duplicates


```python
# number of duplicate rows
num_dups = listings_df.duplicated().sum()
num_dups
```




    0



Next we'll looks at rows or columns containing a lot of missing values. We'll choose a lower threshold of 25% missing values


```python
def prop_missing_vals_df(df, axis=0, threshold=0.25, ascending=False):
    """Get missing values in df by proportion above threshold along axis."""
    prop = df.isna().sum(axis=axis).sort_values(ascending=ascending)
    prop = prop/df.shape[axis]
    prop.name = 'prop_miss_vals'
    return pd.DataFrame(prop[prop > threshold])
```


```python
# rows with high proportion of missing values in listings df
prop_missing_vals_df(listings_df, axis=1, threshold=0.25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>prop_miss_vals</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">vancouver</th>
      <th>3263</th>
      <td>0.323810</td>
    </tr>
    <tr>
      <th>5791</th>
      <td>0.323810</td>
    </tr>
    <tr>
      <th>6006</th>
      <td>0.323810</td>
    </tr>
    <tr>
      <th>5564</th>
      <td>0.323810</td>
    </tr>
    <tr>
      <th>seattle</th>
      <th>7021</th>
      <td>0.323810</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>portland</th>
      <th>4295</th>
      <td>0.257143</td>
    </tr>
    <tr>
      <th>seattle</th>
      <th>764</th>
      <td>0.257143</td>
    </tr>
    <tr>
      <th>portland</th>
      <th>3894</th>
      <td>0.257143</td>
    </tr>
    <tr>
      <th>vancouver</th>
      <th>5921</th>
      <td>0.257143</td>
    </tr>
    <tr>
      <th>seattle</th>
      <th>2987</th>
      <td>0.257143</td>
    </tr>
  </tbody>
</table>
<p>326 rows × 1 columns</p>
</div>



There are quite 326 rows missing more than 25% of values, but no rows are missing 100% of values. We'll address these missing values by column instead.


```python
# columns with high proportion of missing values in listings df
prop_missing_vals_df(listings_df, axis=0, threshold=0.25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prop_miss_vals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>xl_picture_url</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>medium_url</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>thumbnail_url</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>square_feet</th>
      <td>0.963536</td>
    </tr>
    <tr>
      <th>monthly_price</th>
      <td>0.906560</td>
    </tr>
    <tr>
      <th>weekly_price</th>
      <td>0.893266</td>
    </tr>
    <tr>
      <th>neighbourhood_group_cleansed</th>
      <td>0.590645</td>
    </tr>
    <tr>
      <th>notes</th>
      <td>0.414618</td>
    </tr>
    <tr>
      <th>access</th>
      <td>0.340713</td>
    </tr>
    <tr>
      <th>host_about</th>
      <td>0.291551</td>
    </tr>
    <tr>
      <th>transit</th>
      <td>0.258994</td>
    </tr>
  </tbody>
</table>
</div>



Inspecting each of these more carefully, we'll drop all of these columns for the following reasons:

- `transit`, `host_about`, `access`, `notes`: These are all text fields which would require somewhat sophisticated NLP.
- `neighbourhood_group_cleansed`: Large proportion of missing values and we can get more precise information from `neighborhood`
- `weekly_price`, `monthly_price`: Large proportion of missing values and we're more interested in (daily) `price`
- `square_feet`, `thumbnail_url`, `xl_picture_url`, `medium_url`, `host_acceptance_rate`: Too high percentage of missing values to be useful.


```python
# start collecting columns to drop from listings df
drop_cols = set(prop_missing_vals_df(listings_df, axis=0).index)
```

There are quite a few more columns we can drop. To make this easier to manage, lets display columns of `listings` in alphabetical order:


```python
# alphabetize columns besides those selected for dropping and inspect
alph_cols = list(set(listings_df.columns).difference(drop_cols))
alph_cols.sort()
listings_df[alph_cols].head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>accommodates</th>
      <th>amenities</th>
      <th>availability_30</th>
      <th>availability_365</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>bathrooms</th>
      <th>bed_type</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>calendar_last_scraped</th>
      <th>calendar_updated</th>
      <th>cancellation_policy</th>
      <th>cleaning_fee</th>
      <th>country</th>
      <th>country_code</th>
      <th>description</th>
      <th>experiences_offered</th>
      <th>extra_people</th>
      <th>first_review</th>
      <th>guests_included</th>
      <th>has_availability</th>
      <th>host_acceptance_rate</th>
      <th>host_has_profile_pic</th>
      <th>host_id</th>
      <th>host_identity_verified</th>
      <th>host_is_superhost</th>
      <th>host_listings_count</th>
      <th>host_location</th>
      <th>host_name</th>
      <th>host_neighbourhood</th>
      <th>host_picture_url</th>
      <th>host_response_rate</th>
      <th>host_response_time</th>
      <th>host_since</th>
      <th>host_thumbnail_url</th>
      <th>host_total_listings_count</th>
      <th>host_url</th>
      <th>host_verifications</th>
      <th>house_rules</th>
      <th>id</th>
      <th>instant_bookable</th>
      <th>interaction</th>
      <th>is_business_travel_ready</th>
      <th>is_location_exact</th>
      <th>jurisdiction_names</th>
      <th>last_review</th>
      <th>last_scraped</th>
      <th>latitude</th>
      <th>license</th>
      <th>listing_url</th>
      <th>longitude</th>
      <th>market</th>
      <th>maximum_maximum_nights</th>
      <th>maximum_minimum_nights</th>
      <th>maximum_nights</th>
      <th>maximum_nights_avg_ntm</th>
      <th>minimum_maximum_nights</th>
      <th>minimum_minimum_nights</th>
      <th>minimum_nights</th>
      <th>minimum_nights_avg_ntm</th>
      <th>name</th>
      <th>neighborhood_overview</th>
      <th>neighbourhood</th>
      <th>neighbourhood_cleansed</th>
      <th>number_of_reviews</th>
      <th>number_of_reviews_ltm</th>
      <th>picture_url</th>
      <th>price</th>
      <th>property_type</th>
      <th>require_guest_phone_verification</th>
      <th>require_guest_profile_picture</th>
      <th>requires_license</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_checkin</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_rating</th>
      <th>review_scores_value</th>
      <th>reviews_per_month</th>
      <th>room_type</th>
      <th>scrape_id</th>
      <th>security_deposit</th>
      <th>smart_location</th>
      <th>space</th>
      <th>state</th>
      <th>street</th>
      <th>summary</th>
      <th>zipcode</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>seattle</th>
      <th>0</th>
      <td>9</td>
      <td>{Internet,Wifi,Kitchen,"Free parking on premis...</td>
      <td>6</td>
      <td>65</td>
      <td>36</td>
      <td>65</td>
      <td>2.5</td>
      <td>Real Bed</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2020-02-22</td>
      <td>4 weeks ago</td>
      <td>strict_14_with_grace_period</td>
      <td>$250.00</td>
      <td>United States</td>
      <td>US</td>
      <td>Gorgeous, architect remodeled, Dutch Colonial ...</td>
      <td>none</td>
      <td>$25.00</td>
      <td>2008-09-15</td>
      <td>8</td>
      <td>t</td>
      <td>92%</td>
      <td>t</td>
      <td>2536</td>
      <td>f</td>
      <td>t</td>
      <td>2.0</td>
      <td>Seattle, Washington, United States</td>
      <td>Megan</td>
      <td>Minor</td>
      <td>https://a0.muscache.com/im/pictures/user/016a1...</td>
      <td>100%</td>
      <td>within an hour</td>
      <td>2008-08-26</td>
      <td>https://a0.muscache.com/im/pictures/user/016a1...</td>
      <td>2.0</td>
      <td>https://www.airbnb.com/users/show/2536</td>
      <td>['email', 'phone', 'reviews', 'jumio', 'offlin...</td>
      <td>NaN</td>
      <td>2318</td>
      <td>f</td>
      <td>We are a family who live next door and are ava...</td>
      <td>f</td>
      <td>t</td>
      <td>{WASHINGTON," Seattle"," WA"}</td>
      <td>2020-02-01</td>
      <td>2020-02-22</td>
      <td>47.61082</td>
      <td>STR-OPLI-19-002837</td>
      <td>https://www.airbnb.com/rooms/2318</td>
      <td>-122.29082</td>
      <td>Seattle</td>
      <td>1000</td>
      <td>7</td>
      <td>1000</td>
      <td>1000.0</td>
      <td>1000</td>
      <td>7</td>
      <td>7</td>
      <td>7.0</td>
      <td>Casa Madrona - Urban Oasis 1 block from the park!</td>
      <td>Madrona is a hidden gem of a neighborhood. It ...</td>
      <td>Madrona</td>
      <td>Madrona</td>
      <td>32</td>
      <td>11</td>
      <td>https://a0.muscache.com/im/pictures/02973ad3-a...</td>
      <td>$296.00</td>
      <td>House</td>
      <td>f</td>
      <td>f</td>
      <td>t</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>100.0</td>
      <td>10.0</td>
      <td>0.23</td>
      <td>Entire home/apt</td>
      <td>20200222045957</td>
      <td>$500.00</td>
      <td>Seattle, WA</td>
      <td>This beautiful, gracious home has been complet...</td>
      <td>WA</td>
      <td>Seattle, WA, United States</td>
      <td>Gorgeous, architect remodeled, Dutch Colonial ...</td>
      <td>98122</td>
    </tr>
  </tbody>
</table>
</div>



We'll also drop the following columns for the following reasons:

- `interaction`, `neighbourhood_overview`, `description`, `house_rules`, `name`, `space`, `summary`: Long string text columns that could be useful but would require some NLP (which is beyond this scope of this project). 
- `calendar_updated`, `host_location`, `host_name`, `host_neighbourhood`, `host_picture_url`,  `host_thumbnail_url`, `host_url`, `host_verifications`, `is_location_exact`, `jurisdiction_names`, `lisence`, `listing_url`, `market`, `name`, `picture_url`, `scrape_id`: Irrelevant for our analysis
- `maximum_minimum_nights`, `minimum_maximum_nights`, `maximum_maximum_nights`, `minimum_minimum_nights`,  `minimum_nights_avg_ntm`, `maximum_nights_avg_ntm` - Difficult to interpret and possibly less relevant (given presence of `minimum_nights`, `maximum_nights`)
- `last_scraped`, `calendar_last_scraped`, `country`, `country_code`, `experiences_offered`, `has_availability`: All have the same value for this dataset
- `host_has_profile_pic`: Extreme class imbalance (99.8% vs. 0.2%).
- `smart_location`: Highly imbalanced and superceded by other variables.
- `street`: Doesn't even contain street information!
- `state`: Redundant (given city information) and not relevant
- `zipcode`: Superceded by other more useful geographic variables (`neighbourhood`, `latitude`, `longitude`)
- `host_listings_count`, `host_total_listings_count`, `neighbourhood`: Appears to be from original AirBnB data - superceded by other variables created by Inside AirBnB (names beginning    with prefix `calculated_`)
- `number_of_reviews_ltm` - meaning unclear

That was a lot of detailed justification! But it can be helpful to be explicit and clear about reasons for ignoring information.


```python
# add remaining columns to drop and drop
more_drop_cols = {'interaction', 'neighborhood_overview', 'description', 'house_rules', 
                  'name', 'space', 'summary', 'calendar_updated', 'host_location',
                  'host_name', 'host_neighbourhood', 'host_picture_url', 
                  'host_thumbnail_url', 'host_url', 'host_verifications', 
                  'is_location_exact', 'jurisdiction_names', 'listing_url', 'market', 
                  'name', 'picture_url', 'scrape_id', 'maximum_minimum_nights', 
                  'minimum_maximum_nights', 'maximum_maximum_nights', 
                  'minimum_minimum_nights', 'minimum_nights_avg_ntm',
                  'maximum_nights_avg_ntm', 'last_scraped', 'calendar_last_scraped', 
                  'country', 'country_code', 'experiences_offered', 'has_availability', 
                  'host_has_profile_pic', 'smart_location', 'street', 'state', 
                  'host_listings_count', 'host_total_listings_count', 
                  'neighbourhood', 'zipcode', 'number_of_reviews_ltm', 'license'}
drop_cols = drop_cols.union(more_drop_cols)
listings_df = listings_df.drop(columns=drop_cols)
```


```python
def alphabetize_cols(df, first_col='id'):
    """Alphabetize columns."""
    df_cp = df.copy()
    alph_cols = list(df_cp.columns.sort_values())
    alph_cols.remove(first_col)
    alph_cols = [first_col] + alph_cols
    df_cp = df_cp[alph_cols]
    return df_cp

listings_df = alphabetize_cols(listings_df)
```

##### Enforce `listings` dtypes

Now that we're left with the variables we're interested in, we'll see if any need processing before we deal with missing values and setting dtypes


```python
listings_df.select_dtypes('int64').info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 18429 entries, ('seattle', 0) to ('vancouver', 6205)
    Data columns (total 15 columns):
     #   Column                                        Non-Null Count  Dtype
    ---  ------                                        --------------  -----
     0   id                                            18429 non-null  int64
     1   accommodates                                  18429 non-null  int64
     2   availability_30                               18429 non-null  int64
     3   availability_365                              18429 non-null  int64
     4   availability_60                               18429 non-null  int64
     5   availability_90                               18429 non-null  int64
     6   calculated_host_listings_count                18429 non-null  int64
     7   calculated_host_listings_count_entire_homes   18429 non-null  int64
     8   calculated_host_listings_count_private_rooms  18429 non-null  int64
     9   calculated_host_listings_count_shared_rooms   18429 non-null  int64
     10  guests_included                               18429 non-null  int64
     11  host_id                                       18429 non-null  int64
     12  maximum_nights                                18429 non-null  int64
     13  minimum_nights                                18429 non-null  int64
     14  number_of_reviews                             18429 non-null  int64
    dtypes: int64(15)
    memory usage: 2.2+ MB


All of the variables with integer type look good. Also since `np.nan` is a float, we can conclude all these variables have no missing values


```python
listings_df.select_dtypes('float64').info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 18429 entries, ('seattle', 0) to ('vancouver', 6205)
    Data columns (total 13 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   bathrooms                    18424 non-null  float64
     1   bedrooms                     18409 non-null  float64
     2   beds                         18383 non-null  float64
     3   latitude                     18429 non-null  float64
     4   longitude                    18429 non-null  float64
     5   review_scores_accuracy       15990 non-null  float64
     6   review_scores_checkin        15989 non-null  float64
     7   review_scores_cleanliness    15990 non-null  float64
     8   review_scores_communication  15993 non-null  float64
     9   review_scores_location       15990 non-null  float64
     10  review_scores_rating         15999 non-null  float64
     11  review_scores_value          15991 non-null  float64
     12  reviews_per_month            16089 non-null  float64
    dtypes: float64(13)
    memory usage: 1.9+ MB


All of these variables are indeed numerical. Some of them could be converted to ints -- this may speed computations and reduce the chance of rounding errors. Since `np.nan` is a float however, we'll wait until after we handle missing values. Let's look at which floats can be safely converted to ints.


```python
def can_conv_to_int(list_float_vars):
    """Check which float values can be converted to int without rounding."""
    res = (np.abs(np.floor(list_float_vars) - list_float_vars) > 0).any()
    return ~ res
```


```python
# dictionary for tracking col dtype conversions
conv_dtypes = defaultdict(set)
# see which float columns can be safely converted to int
conv_to_int = can_conv_to_int(listings_df.select_dtypes('float64').dropna())
conv_to_int_cols = set(conv_to_int[conv_to_int.T].index)
conv_dtypes['int'] = conv_to_int_cols
conv_dtypes['int']
```




    {'bedrooms',
     'beds',
     'review_scores_accuracy',
     'review_scores_checkin',
     'review_scores_cleanliness',
     'review_scores_communication',
     'review_scores_location',
     'review_scores_rating',
     'review_scores_value'}



So we can safely convert all but `reviews_per_month`, `longitude`, `latitude`, and `bathrooms` to integer type (after we've dealt with missing values)

Finally let's look at object variables


```python
listings_df.select_dtypes('object').info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 18429 entries, ('seattle', 0) to ('vancouver', 6205)
    Data columns (total 23 columns):
     #   Column                            Non-Null Count  Dtype 
    ---  ------                            --------------  ----- 
     0   amenities                         18429 non-null  object
     1   bed_type                          18429 non-null  object
     2   cancellation_policy               18429 non-null  object
     3   cleaning_fee                      17148 non-null  object
     4   extra_people                      18429 non-null  object
     5   first_review                      16089 non-null  object
     6   host_acceptance_rate              17008 non-null  object
     7   host_identity_verified            18428 non-null  object
     8   host_is_superhost                 18428 non-null  object
     9   host_response_rate                15535 non-null  object
     10  host_response_time                15535 non-null  object
     11  host_since                        18428 non-null  object
     12  instant_bookable                  18429 non-null  object
     13  is_business_travel_ready          18429 non-null  object
     14  last_review                       16089 non-null  object
     15  neighbourhood_cleansed            18429 non-null  object
     16  price                             18429 non-null  object
     17  property_type                     18429 non-null  object
     18  require_guest_phone_verification  18429 non-null  object
     19  require_guest_profile_picture     18429 non-null  object
     20  requires_license                  18429 non-null  object
     21  room_type                         18429 non-null  object
     22  security_deposit                  15649 non-null  object
    dtypes: object(23)
    memory usage: 3.3+ MB



```python
listings_df.select_dtypes('object').head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>amenities</th>
      <th>bed_type</th>
      <th>cancellation_policy</th>
      <th>cleaning_fee</th>
      <th>extra_people</th>
      <th>first_review</th>
      <th>host_acceptance_rate</th>
      <th>host_identity_verified</th>
      <th>host_is_superhost</th>
      <th>host_response_rate</th>
      <th>host_response_time</th>
      <th>host_since</th>
      <th>instant_bookable</th>
      <th>is_business_travel_ready</th>
      <th>last_review</th>
      <th>neighbourhood_cleansed</th>
      <th>price</th>
      <th>property_type</th>
      <th>require_guest_phone_verification</th>
      <th>require_guest_profile_picture</th>
      <th>requires_license</th>
      <th>room_type</th>
      <th>security_deposit</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>seattle</th>
      <th>0</th>
      <td>{Internet,Wifi,Kitchen,"Free parking on premis...</td>
      <td>Real Bed</td>
      <td>strict_14_with_grace_period</td>
      <td>$250.00</td>
      <td>$25.00</td>
      <td>2008-09-15</td>
      <td>92%</td>
      <td>f</td>
      <td>t</td>
      <td>100%</td>
      <td>within an hour</td>
      <td>2008-08-26</td>
      <td>f</td>
      <td>f</td>
      <td>2020-02-01</td>
      <td>Madrona</td>
      <td>$296.00</td>
      <td>House</td>
      <td>f</td>
      <td>f</td>
      <td>t</td>
      <td>Entire home/apt</td>
      <td>$500.00</td>
    </tr>
  </tbody>
</table>
</div>



Some of these variables can be converted to other appropriate types, without additional processing. We're careful not to cast to boolean dtype with missing data since pandas will automatically convert `np.nan` to `True`


```python
# non-ordered categorical variables
conv_dtypes['categorical'] = {'host_identity_verified', 'host_is_superhost', 'instant_bookable', 
                              'is_business_travel_ready', 'neighbourhood_cleansed', 'property_type', 
                              'require_guest_phone_verification', 'require_guest_profile_picture', 
                              'requires_license'}
conv_dtypes['datetime'] = {'first_review', 'host_since', 'last_review'}

listings_df.loc[:, conv_dtypes['categorical']] = listings_df[conv_dtypes[
                                                 'categorical']].astype('category')
listings_df.loc[:, conv_dtypes['datetime']] = listings_df[conv_dtypes['datetime']]\
                                           .astype('datetime64')
```

We also have a some categorical data types with natural orderings, since there's a natural way of assigning a price value to them.


```python
def set_ord_cat_dtypes(listings_df, conv_dtypes, ord_cat_cols):
    """Set ordered categorical columns dtypes."""
    listings_df_cp = listings_df.copy()
    for (col, ordering) in conv_dtypes['ordered_categorical'].items():
        dtype = CategoricalDtype(categories=ordering, ordered=True)
        listings_df_cp[col] = listings_df_cp[col].astype(dtype)
    return listings_df_cp
```


```python
# ordered categorical variables
ord_cat_cols = ['bed_type', 'cancellation_policy', 'host_response_time', 'room_type']
conv_dtypes['ordered_categorical'] = {col:None for col in ord_cat_cols}
conv_dtypes['ordered_categorical']['bed_type'] = ['Couch', 'Airbed', 'Pull-out Sofa', 
                                                  'Futon', 'Real Bed']
conv_dtypes['ordered_categorical']['cancellation_policy'] = ['super_strict_60', 
                                                             'super_strict_30', 
                                                             'strict', 
                                                             'strict_14_with_grace_period', 
                                                             'moderate', 'flexible']
conv_dtypes['ordered_categorical']['host_response_time'] = ['within an hour', 
                                                            'within a few hours', 
                                                            'within a day', 
                                                            'a few days of more']
conv_dtypes['ordered_categorical']['room_type'] = ['Shared room', 'Hotel room', 
                                                   'Private room', 'Entire home/apt']
listings_df = set_ord_cat_dtypes(listings_df, conv_dtypes, ord_cat_cols)
```

Finally, some of the object variables need to be processed. Some can be easily converted to floats -- in this case, it doesn't matter if `np.nan` is present or not.


```python
def conv_to_float(entry):
    """Float conversion helper."""
    try:
        return entry.replace('$', '').replace('%', '').replace(',', '')
    except AttributeError:
        return entry
    
def conv_cols(listings_df, conv_dtypes, conv_func, dtype_name):
    """Process/convert columns."""
    listings_df_cp = listings_df.copy()

    for col in conv_dtypes[dtype_name]:
        listings_df_cp[col] = listings_df[col].apply(conv_func)\
                              .astype(dtype_name)

    return listings_df_cp
```


```python
# some float variables
conv_dtypes['float'] = {'cleaning_fee', 'extra_people', 'price', 
                        'security_deposit', 'host_response_rate', 
                        'host_acceptance_rate'}
listings_df = conv_cols(listings_df, conv_dtypes, conv_to_float, 'float')
```

##### Missing `listings` values

Now we'll take a look at missing values


```python
# columns in listings dataset missing values
miss_vals_df = prop_missing_vals_df(listings_df, axis=0, threshold=0)
miss_vals_df = pd.DataFrame({'col': miss_vals_df.index, 
                            'prop_miss': miss_vals_df['prop_miss_vals'].values, 
                             'dtype': listings_df[miss_vals_df.index].dtypes})
miss_vals_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col</th>
      <th>prop_miss</th>
      <th>dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>host_response_time</th>
      <td>host_response_time</td>
      <td>0.165446</td>
      <td>category</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>host_response_rate</td>
      <td>0.157035</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>security_deposit</th>
      <td>security_deposit</td>
      <td>0.150849</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_checkin</th>
      <td>review_scores_checkin</td>
      <td>0.132400</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_location</th>
      <td>review_scores_location</td>
      <td>0.132346</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_cleanliness</th>
      <td>review_scores_cleanliness</td>
      <td>0.132346</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_accuracy</th>
      <td>review_scores_accuracy</td>
      <td>0.132346</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_value</th>
      <td>review_scores_value</td>
      <td>0.132291</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_communication</th>
      <td>review_scores_communication</td>
      <td>0.132183</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>review_scores_rating</td>
      <td>0.131857</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>reviews_per_month</th>
      <td>reviews_per_month</td>
      <td>0.126974</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>last_review</th>
      <td>last_review</td>
      <td>0.126974</td>
      <td>datetime64[ns]</td>
    </tr>
    <tr>
      <th>first_review</th>
      <td>first_review</td>
      <td>0.126974</td>
      <td>datetime64[ns]</td>
    </tr>
    <tr>
      <th>host_acceptance_rate</th>
      <td>host_acceptance_rate</td>
      <td>0.077107</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>cleaning_fee</th>
      <td>cleaning_fee</td>
      <td>0.069510</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>beds</td>
      <td>0.002496</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>bedrooms</td>
      <td>0.001085</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>bathrooms</td>
      <td>0.000271</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>host_identity_verified</th>
      <td>host_identity_verified</td>
      <td>0.000054</td>
      <td>category</td>
    </tr>
    <tr>
      <th>host_since</th>
      <td>host_since</td>
      <td>0.000054</td>
      <td>datetime64[ns]</td>
    </tr>
    <tr>
      <th>host_is_superhost</th>
      <td>host_is_superhost</td>
      <td>0.000054</td>
      <td>category</td>
    </tr>
  </tbody>
</table>
</div>



First focussing on review-related variables missing variables, we notice the proportion of missing values is very close.


```python
# review related variables missing values
miss_val_rev_cols = [col for col in miss_vals_df.index if 'review' in col]
miss_vals_df.loc[miss_val_rev_cols, :]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col</th>
      <th>prop_miss</th>
      <th>dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>review_scores_checkin</th>
      <td>review_scores_checkin</td>
      <td>0.132400</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_location</th>
      <td>review_scores_location</td>
      <td>0.132346</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_cleanliness</th>
      <td>review_scores_cleanliness</td>
      <td>0.132346</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_accuracy</th>
      <td>review_scores_accuracy</td>
      <td>0.132346</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_value</th>
      <td>review_scores_value</td>
      <td>0.132291</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_communication</th>
      <td>review_scores_communication</td>
      <td>0.132183</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>review_scores_rating</td>
      <td>0.131857</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>reviews_per_month</th>
      <td>reviews_per_month</td>
      <td>0.126974</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>last_review</th>
      <td>last_review</td>
      <td>0.126974</td>
      <td>datetime64[ns]</td>
    </tr>
    <tr>
      <th>first_review</th>
      <td>first_review</td>
      <td>0.126974</td>
      <td>datetime64[ns]</td>
    </tr>
  </tbody>
</table>
</div>



We're both interested in the review variables as responses varialbes, so imputing is problematic, especially given the high percentage of missing values. Moreover, it's possible that the some listings are missing values for all review variables.


```python
# prop of listings missing all review variables
miss_val_rev_df = listings_df[listings_df[miss_val_rev_cols].isna().T.all()]\
                            [miss_val_rev_cols]
len(miss_val_rev_df)/len(listings_df)
```




    0.1269737913071789



So almost all of the variables missing some review information are missing all review information. Let's compare summary statistics for the listings missing all review information to summary statistics for the entire dataset.


```python
# summary statistics for listings missing review variable values
listings_df.loc[miss_val_rev_df.index].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>accommodates</th>
      <th>availability_30</th>
      <th>availability_365</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>cleaning_fee</th>
      <th>extra_people</th>
      <th>guests_included</th>
      <th>host_acceptance_rate</th>
      <th>host_id</th>
      <th>host_response_rate</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>maximum_nights</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>price</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_checkin</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_rating</th>
      <th>review_scores_value</th>
      <th>reviews_per_month</th>
      <th>security_deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.340000e+03</td>
      <td>2340.000000</td>
      <td>2340.000000</td>
      <td>2340.000000</td>
      <td>2340.000000</td>
      <td>2340.000000</td>
      <td>2339.000000</td>
      <td>2331.000000</td>
      <td>2300.000000</td>
      <td>2340.000000</td>
      <td>2340.000000</td>
      <td>2340.000000</td>
      <td>2340.000000</td>
      <td>1878.000000</td>
      <td>2340.000000</td>
      <td>2340.000000</td>
      <td>1865.000000</td>
      <td>2.340000e+03</td>
      <td>1811.000000</td>
      <td>2340.000000</td>
      <td>2340.000000</td>
      <td>2340.000000</td>
      <td>2340.000000</td>
      <td>2340.0</td>
      <td>2340.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1734.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.414823e+07</td>
      <td>3.338889</td>
      <td>14.158547</td>
      <td>157.964103</td>
      <td>31.888889</td>
      <td>50.710256</td>
      <td>1.340744</td>
      <td>1.422136</td>
      <td>1.717826</td>
      <td>51.204274</td>
      <td>47.062821</td>
      <td>1.230769</td>
      <td>0.525641</td>
      <td>111.756656</td>
      <td>10.141026</td>
      <td>1.678205</td>
      <td>85.252011</td>
      <td>1.093410e+08</td>
      <td>94.926560</td>
      <td>47.739875</td>
      <td>-122.692045</td>
      <td>777.929487</td>
      <td>20.210684</td>
      <td>0.0</td>
      <td>218.774359</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>466.691465</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.352976e+06</td>
      <td>2.105213</td>
      <td>12.568005</td>
      <td>129.301803</td>
      <td>24.766538</td>
      <td>36.762457</td>
      <td>0.645418</td>
      <td>1.008679</td>
      <td>1.257231</td>
      <td>95.190725</td>
      <td>95.449701</td>
      <td>2.958844</td>
      <td>4.424825</td>
      <td>101.441680</td>
      <td>19.850892</td>
      <td>1.424858</td>
      <td>25.003858</td>
      <td>1.130969e+08</td>
      <td>16.545621</td>
      <td>1.398476</td>
      <td>0.345667</td>
      <td>530.030688</td>
      <td>30.511747</td>
      <td>0.0</td>
      <td>512.201897</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>783.654495</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.814600e+04</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.014300e+04</td>
      <td>0.000000</td>
      <td>45.436220</td>
      <td>-123.218790</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.267230e+07</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>43.750000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>50.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>68.000000</td>
      <td>8.534462e+06</td>
      <td>99.000000</td>
      <td>47.547308</td>
      <td>-123.106838</td>
      <td>90.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>89.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.707740e+07</td>
      <td>2.000000</td>
      <td>14.000000</td>
      <td>126.000000</td>
      <td>38.000000</td>
      <td>63.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>89.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>4.880712e+07</td>
      <td>100.000000</td>
      <td>47.636640</td>
      <td>-122.673940</td>
      <td>1125.000000</td>
      <td>30.000000</td>
      <td>0.0</td>
      <td>142.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>300.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.094748e+07</td>
      <td>4.000000</td>
      <td>28.000000</td>
      <td>274.000000</td>
      <td>58.000000</td>
      <td>88.000000</td>
      <td>1.500000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>28.000000</td>
      <td>13.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>145.000000</td>
      <td>20.000000</td>
      <td>2.000000</td>
      <td>100.000000</td>
      <td>2.225925e+08</td>
      <td>100.000000</td>
      <td>49.253462</td>
      <td>-122.337650</td>
      <td>1125.000000</td>
      <td>30.000000</td>
      <td>0.0</td>
      <td>200.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.244232e+07</td>
      <td>20.000000</td>
      <td>30.000000</td>
      <td>365.000000</td>
      <td>60.000000</td>
      <td>90.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>25.000000</td>
      <td>57.000000</td>
      <td>656.000000</td>
      <td>350.000000</td>
      <td>20.000000</td>
      <td>100.000000</td>
      <td>3.368498e+08</td>
      <td>100.000000</td>
      <td>49.293880</td>
      <td>-122.240950</td>
      <td>9999.000000</td>
      <td>365.000000</td>
      <td>0.0</td>
      <td>9999.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6709.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# summary statistics for full listings dataset
listings_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>accommodates</th>
      <th>availability_30</th>
      <th>availability_365</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>cleaning_fee</th>
      <th>extra_people</th>
      <th>guests_included</th>
      <th>host_acceptance_rate</th>
      <th>host_id</th>
      <th>host_response_rate</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>maximum_nights</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>price</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_checkin</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_rating</th>
      <th>review_scores_value</th>
      <th>reviews_per_month</th>
      <th>security_deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.842900e+04</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>18424.000000</td>
      <td>18409.000000</td>
      <td>18383.000000</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>17148.000000</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>17008.000000</td>
      <td>1.842900e+04</td>
      <td>15535.000000</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>18429.000000</td>
      <td>15990.000000</td>
      <td>15989.000000</td>
      <td>15990.000000</td>
      <td>15993.000000</td>
      <td>15990.00000</td>
      <td>15999.000000</td>
      <td>15991.000000</td>
      <td>16089.000000</td>
      <td>15649.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.389069e+07</td>
      <td>3.568126</td>
      <td>12.350426</td>
      <td>148.143307</td>
      <td>28.582506</td>
      <td>46.318248</td>
      <td>1.295538</td>
      <td>1.417622</td>
      <td>1.858075</td>
      <td>17.553476</td>
      <td>15.576591</td>
      <td>0.824896</td>
      <td>0.252700</td>
      <td>80.080651</td>
      <td>13.715448</td>
      <td>1.970644</td>
      <td>91.943438</td>
      <td>8.178062e+07</td>
      <td>97.533891</td>
      <td>47.643656</td>
      <td>-122.676035</td>
      <td>612.999891</td>
      <td>11.241250</td>
      <td>55.183081</td>
      <td>159.907483</td>
      <td>9.766354</td>
      <td>9.826756</td>
      <td>9.655159</td>
      <td>9.834615</td>
      <td>9.81207</td>
      <td>95.576224</td>
      <td>9.540742</td>
      <td>2.411813</td>
      <td>306.152662</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.238720e+07</td>
      <td>2.157336</td>
      <td>11.021889</td>
      <td>121.067805</td>
      <td>22.045518</td>
      <td>32.808867</td>
      <td>0.628650</td>
      <td>1.019990</td>
      <td>1.367288</td>
      <td>56.309083</td>
      <td>55.606749</td>
      <td>2.218354</td>
      <td>3.339698</td>
      <td>67.365202</td>
      <td>23.227078</td>
      <td>1.584518</td>
      <td>17.186222</td>
      <td>9.396449e+07</td>
      <td>10.668785</td>
      <td>1.420644</td>
      <td>0.336688</td>
      <td>559.068779</td>
      <td>23.188497</td>
      <td>82.727205</td>
      <td>258.998090</td>
      <td>0.662001</td>
      <td>0.600856</td>
      <td>0.733419</td>
      <td>0.607892</td>
      <td>0.53100</td>
      <td>6.720840</td>
      <td>0.761208</td>
      <td>2.201809</td>
      <td>515.534116</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.318000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.618000e+03</td>
      <td>0.000000</td>
      <td>45.434560</td>
      <td>-123.218790</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.00000</td>
      <td>20.000000</td>
      <td>2.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.410369e+07</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>44.000000</td>
      <td>3.000000</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>93.000000</td>
      <td>8.534462e+06</td>
      <td>100.000000</td>
      <td>45.591890</td>
      <td>-123.088580</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>76.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>10.000000</td>
      <td>10.00000</td>
      <td>94.000000</td>
      <td>9.000000</td>
      <td>0.620000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.464953e+07</td>
      <td>3.000000</td>
      <td>11.000000</td>
      <td>123.000000</td>
      <td>30.000000</td>
      <td>51.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>60.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>3.836791e+07</td>
      <td>100.000000</td>
      <td>47.634460</td>
      <td>-122.637710</td>
      <td>1124.000000</td>
      <td>2.000000</td>
      <td>21.000000</td>
      <td>113.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.00000</td>
      <td>98.000000</td>
      <td>10.000000</td>
      <td>1.790000</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.516444e+07</td>
      <td>4.000000</td>
      <td>23.000000</td>
      <td>252.000000</td>
      <td>49.000000</td>
      <td>77.000000</td>
      <td>1.500000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>20.000000</td>
      <td>2.000000</td>
      <td>100.000000</td>
      <td>1.342970e+08</td>
      <td>100.000000</td>
      <td>49.248060</td>
      <td>-122.341340</td>
      <td>1125.000000</td>
      <td>30.000000</td>
      <td>73.000000</td>
      <td>180.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.00000</td>
      <td>99.000000</td>
      <td>10.000000</td>
      <td>3.670000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.244232e+07</td>
      <td>26.000000</td>
      <td>30.000000</td>
      <td>365.000000</td>
      <td>60.000000</td>
      <td>90.000000</td>
      <td>16.000000</td>
      <td>30.000000</td>
      <td>49.000000</td>
      <td>302.000000</td>
      <td>302.000000</td>
      <td>25.000000</td>
      <td>57.000000</td>
      <td>700.000000</td>
      <td>399.000000</td>
      <td>20.000000</td>
      <td>100.000000</td>
      <td>3.368498e+08</td>
      <td>100.000000</td>
      <td>49.293880</td>
      <td>-122.240950</td>
      <td>10000.000000</td>
      <td>998.000000</td>
      <td>828.000000</td>
      <td>13000.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.00000</td>
      <td>100.000000</td>
      <td>10.000000</td>
      <td>16.650000</td>
      <td>6709.000000</td>
    </tr>
  </tbody>
</table>
</div>



The summary statistics suggest that, for listings missing review information the distributions of variables are different, than the distributions for all listings, so dropping these listings could bias the results. However, our analysis focuses on review variables, and moreover since they will be used as response variables, imputation seems like a bad plan. We'll go ahead drop rows missing any review variables


```python
# drop rows missing any review variables
mask = listings_df[miss_val_rev_cols].isna().sum(axis=1) == 0
listings_df = listings_df[mask]
```

We'll also drop `last_review` and `first_review` columns as these are probably not too relevant for our analysis (although `number_of_reviews` and `reviews_per_month` will be)


```python
# drop two more review columns
listings_df = listings_df.drop(columns=['first_review', 'last_review'])
```

Now focusing on the variables `bedrooms`, `bathrooms`, `beds`, `host_since`, `host_identity_verified`, `host_is_superhost`, we notice all have less than 1% missing values, so simple imputation choices seem unproblematic here. Let's look at their distributions.


```python
# quantitative variables with small proportion of missing values
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
for (i, col) in enumerate(['bathrooms', 'beds', 'bedrooms']):
    sns.distplot(listings_df[col].dropna(), ax=axs[i], kde=False)
fig.tight_layout()
```


![png]({{site.baseurl}}/assets/images/wrangling_71_0.png)


Each of these variables is highly peaked and highly right skewed, so the imputing them with the mode (instead of mean or median) is a reasonable choice


```python
# categorical variables with small proportion of missing values
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
for (i, col) in enumerate(['host_is_superhost', 'host_identity_verified']):
    sns.countplot(listings_df[col].dropna(), ax=axs[i % 2])
```


![png]({{site.baseurl}}/assets/images/wrangling_73_0.png)


These binary variables are fairly evenly distributed, so the imputing them with mode is a reasonable choice.


```python
# dictionary for imputation methods for columns with missing values
impute_vals = defaultdict(None)
# imputation values for some columns
for col in ['bathrooms', 'beds', 'bedrooms', 'host_since',
            'host_is_superhost', 'host_identity_verified']:
    impute_vals[col] = listings_df[col].mode().values[0]
```

Now let's look at the distributions of `host_acceptance_rate`, `cleaning_fee` and  `security_deposit`


```python
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
for (i, col) in enumerate(['host_acceptance_rate', 'cleaning_fee', 'security_deposit']):
    sns.distplot(listings_df[col].dropna(), kde=False, norm_hist=True, ax=axs[i])
```


![png]({{site.baseurl}}/assets/images/wrangling_77_0.png)


Given the large peak around zero for `security_deposit` and around 100 for `host_acceptance_rate`, and the the fact that it seems reasonable to assign a value of zero to a listing that doesn't have security deposit information available, we'll use this for our imputation value. We'll use median for `cleaning_fee`, since it's highly peaked and skewed at a non-zero value.


```python
# imputation values for more columns
impute_vals['security_deposit'] = 0
impute_vals['cleaning_fee'] = listings_df['cleaning_fee'].dropna().median()
impute_vals['host_acceptance_rate'] = listings_df['host_acceptance_rate'].dropna().median()
```

Finally let's look at the distributions of `host_response_rate` and `host_response_time`


```python
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
sns.distplot(listings_df['host_response_rate'].dropna(), kde=False, norm_hist=True, ax=axs[0])
sns.countplot(listings_df['host_response_time'].dropna(), ax=axs[1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x14362b280>




![png]({{site.baseurl}}/assets/images/wrangling_81_1.png)


Given the large skewness and peaks, we'll impute these with mode as well


```python
# last imputation values
for col in ['host_response_rate', 'host_response_time']:
    impute_vals[col] = listings_df[col].mode().values[0]
```


```python
# impute all missing values
listings_df = listings_df.fillna(impute_vals)
```

##### Downcast `listings` dtypes

In the last section we noted that some float variables could be downcast to ints to speed up computation. We'll do that now.


```python
# downcast to int
listings_df.loc[:, conv_dtypes['int']] = listings_df[conv_dtypes['int']].astype('int') 
```

Now that we have no missing values, we can also cast some categorical to boolean dtypes


```python
def conv_to_bool(entry):
    """Convert string entries to booleans."""
    return entry == 't'

conv_dtypes['bool'] = {'host_identity_verified', 'host_is_superhost', 
                       'instant_bookable', 'is_business_travel_ready', 
                       'require_guest_phone_verification',
                       'require_guest_profile_picture', 'requires_license'}

# downcast to boolean
listings_df = conv_cols(listings_df, conv_dtypes, conv_to_bool, 'bool') 
```

#### `calendar` dataset

##### Drop `calendar` rows or columns

We check for duplicates


```python
num_dups = calendar_df.duplicated().sum()
num_dups
```




    0



Next we'll looks at rows or columns containing a lot of missing values. We'll choose a lower threshold of 25% missing values


```python
# rows with high proportion of missing values in listings df
prop_missing_vals_df(calendar_df, axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>prop_miss_vals</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="11" valign="top">vancouver</th>
      <th>175625</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>463</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>461</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>460</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>459</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>197337</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>197336</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>197335</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>197353</th>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>197351</th>
      <td>0.285714</td>
    </tr>
  </tbody>
</table>
<p>1084 rows × 1 columns</p>
</div>




```python
# columns with high proportion of missing values in listings df
prop_missing_vals_df(calendar_df, axis=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prop_miss_vals</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



No columns are missing >25% of values, but 1084 rows are missing about 28.5% of values. We'll look closer at these rows.


```python
# rows missing values
miss_vals_df = calendar_df.loc[calendar_df.isna().sum(axis=1) > 0]
miss_vals_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 1084 entries, ('portland', 298877) to ('vancouver', 1856971)
    Data columns (total 7 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   listing_id      1084 non-null   int64  
     1   date            1084 non-null   object 
     2   available       1084 non-null   object 
     3   price           5 non-null      object 
     4   adjusted_price  5 non-null      object 
     5   minimum_nights  1079 non-null   float64
     6   maximum_nights  1079 non-null   float64
    dtypes: float64(2), int64(1), object(4)
    memory usage: 21.1+ MB



```python
# listing ids of calendar entries with missing values
miss_cal_idx = miss_vals_df.reset_index(level=0)\
               .groupby(['city', 'listing_id'])['listing_id'].count()
miss_cal_idx
```




    city       listing_id
    portland   6132497         5
    vancouver  906964        178
               5807056       178
               7150429       179
               14237741        8
               27466923      179
               38825501      178
               39093654      179
    Name: listing_id, dtype: int64



The missing calendar rows are concentrated in just eight listing ids. Six of these listing ids are missing a large number of entries (around half a year of data), we'll drop these from both data sets. 


```python
# drop listing ids with large number of missing entries
listing_ids = {index[1] for index in miss_cal_idx[miss_cal_idx > 8].index}
list_ids_mask = ~ listings_df['id'].apply(lambda x: x in listing_ids)
listings_df = listings_df[list_ids_mask]
cal_ids_mask = ~ calendar_df['listing_id'].apply(lambda x: x in listing_ids)
calendar_df = calendar_df[cal_ids_mask]
```

We'll also drop `adjusted_price`. This appears to be a variable added by InsideAirBnB, but couldn't find any information about it. Furthermore, we assume `price` in the `calendar` dataset is the equivalent to `price` in `listings`


```python
calendar_df = calendar_df.drop(columns=['adjusted_price'])
```

##### Enforce `calendar` dtypes


```python
calendar_df.dtypes
```




    listing_id          int64
    date               object
    available          object
    price              object
    minimum_nights    float64
    maximum_nights    float64
    dtype: object




```python
# set dtypes
calendar_df.loc[:, 'date'] = calendar_df['date'].astype('datetime64')
conv_dtypes = defaultdict(set)
conv_dtypes['bool'] = {'available'}
conv_dtypes['float'] = {'price'}
calendar_df = conv_cols(calendar_df, conv_dtypes, conv_to_bool, 'bool')
calendar_df = conv_cols(calendar_df, conv_dtypes, conv_to_float, 'float')
```

##### Missing `calendar` values

Finally we'll deal with the remaining missing values in `calendar`


```python
miss_vals_df = calendar_df.loc[calendar_df.isna().sum(axis=1) > 0]
miss_vals_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">vancouver</th>
      <th>398064</th>
      <td>14237741</td>
      <td>2021-02-07</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>398065</th>
      <td>14237741</td>
      <td>2021-02-08</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>398066</th>
      <td>14237741</td>
      <td>2021-02-09</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>398067</th>
      <td>14237741</td>
      <td>2021-02-10</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>398068</th>
      <td>14237741</td>
      <td>2021-02-11</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>398069</th>
      <td>14237741</td>
      <td>2021-02-12</td>
      <td>False</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>398070</th>
      <td>14237741</td>
      <td>2021-02-13</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>398071</th>
      <td>14237741</td>
      <td>2021-02-14</td>
      <td>False</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>30.0</td>
    </tr>
  </tbody>
</table>
</div>



To deal with the missing `minimum_nights` and `maximum_nights` for listing 6132497, we'll look for other values of these variables for this listing.


```python
# unique values of maximum nights
calendar_df[calendar_df['listing_id'] == 6132497]['maximum_nights'].unique()
```




    array([28., nan])




```python
# unique values of maximum nights
calendar_df[calendar_df['listing_id'] == 6132497]['minimum_nights'].unique()
```




    array([ 1., nan])



Since this listing only ever had one non-null value for these variables, it makes sense to impute that


```python
# fill missing values for listing 6132497
miss_cal_idx = miss_vals_df[miss_vals_df['listing_id'] == 6132497].index
calendar_df.loc[miss_cal_idx, 'minimum_nights'] = 1.0
calendar_df.loc[miss_cal_idx, 'maximum_nights'] = 28.0
```

The remaining missing values for listing 14237741 are all in the same week. It makes sense to backfill with prices from the corresponding days on the previous week.


```python
def backfill_missing_prices(calendar_df, miss_price_vals_df):
    """Backfill missing price data for two listings in calendar dataset."""
    week_delta = pd.Timedelta(1, unit='w')
    calendar_df_cp = calendar_df.copy()
    for index in miss_price_vals_df.index:
        listing_id = miss_price_vals_df.loc[index]['listing_id']
        one_week_ago = miss_price_vals_df.loc[index]['date'] - week_delta
        mask = ((calendar_df_cp['listing_id'] == listing_id) & 
               (calendar_df_cp['date'] == one_week_ago))
        price = calendar_df_cp[mask]['price'].values[0]
        calendar_df_cp.loc[index, 'price'] = price
    return calendar_df_cp

miss_price_vals_df = miss_vals_df[miss_vals_df['listing_id'] == 14237741]
calendar_df = backfill_missing_prices(calendar_df, miss_price_vals_df)
```


```python
# check backfill worked
len(calendar_df[(cal_id_mask) &  calendar_df['price'].isna()])
```




    0



##### Synchronize `calendar.date`

The calendar data for the three cities was collected at different times, so we need to sync up the dates for comparison.


```python
maxmin_date = '2020-02-22'
minmax_date = '2021-02-13'
cal_date_mask = ((calendar_df['date'] >= maxmin_date) & (calendar_df['date'] <= minmax_date))
calendar_df = calendar_df[cal_date_mask]
```

### Create amenities features.

There's a lot of potentially useful information in the `amentities` column but it's going to take a little bit of work to get it. Currently the entries in the amenities columns are long strings containing amenities for that listing.


```python
amenities_series = listings_df['amenities']
amenities_series
```




    city           
    seattle    0       {Internet,Wifi,Kitchen,"Free parking on premis...
               1       {TV,Internet,Wifi,"Air conditioning",Kitchen,"...
               2       {Internet,Wifi,"Air conditioning",Kitchen,"Fre...
               3       {TV,"Cable TV",Wifi,"Air conditioning",Kitchen...
               4       {TV,"Cable TV",Internet,Wifi,"Wheelchair acces...
                                             ...                        
    vancouver  6130    {Wifi,Kitchen,"Free street parking",Heating,Wa...
               6145    {TV,Wifi,Kitchen,"Paid parking off premises","...
               6150    {TV,Wifi,"Air conditioning",Pool,Kitchen,Gym,E...
               6152    {TV,"Cable TV",Wifi,Kitchen,"Free parking on p...
               6174    {Wifi,Kitchen,Breakfast,"Free street parking",...
    Name: amenities, Length: 15983, dtype: object



To proceed we need to do a bit of processing.


```python
def process_amenities(amenities_series):
    """Process entries in amenities series."""
    # convert amenities lists into sets of strings
    amenities_series = amenities_series.apply(lambda x: set(x.split(',')))
    # set for tracking unique amenities
    amenities_set = set()
    # for dropping all non alphanumeric characters
    regex = re.compile('[^0-9a-zA-Z ]+')
    # function for processing each amenity lists in series entries
    def process_and_add(amenities_set, regex, amens_entry):
        new_amens_entry = set()
        for amen in amens_entry:
            # drop non alpha numeric
            amen = regex.sub('', amen)
            # don't keep track of translation failures
            if 'translation' in amen:
                pass
            else:
                new_amens_entry.add(amen)
                # add to main set if it hasn't been seen
                if amen not in amenities_set:
                    amenities_set.add(amen)
        return new_amens_entry
    # process amenity list entries in series
    apply_func = partial(process_and_add, amenities_set, regex)
    amenities_series = amenities_series.apply(apply_func)

    return amenities_series, amenities_set


def rename_amenities(amenities_series, amenities_set, amenities_mapping):
    """Clean up amenities names."""
    amenities_set = {amen if amen not in amenities_mapping
                     else amenities_mapping[amen] for amen in amenities_set}
    amenities_set = {'amen_' + amen.replace(' ', '_').lower() for amen
                     in amenities_set}

    # function for renaming amenity lists in series entries
    def rename_amens(amenities_mapping, amens_entry):
        new_amens_entry = set()
        for amen in amens_entry:
            try:
                amen = amenities_mapping[amen]
            except KeyError:
                pass
            amen = 'amen_' + amen.replace(' ', '_').lower()
            new_amens_entry.add(amen)
        return new_amens_entry

    # process amenity list entries in series
    apply_func = partial(rename_amens, amenities_mapping)
    amenities_series = amenities_series.apply(apply_func)

    return amenities_series, amenities_set
```


```python
# get renamed series and set of amenities
amenities_series, amenities_set = process_amenities(
                                      listings_df['amenities'])
amenities_mapping = {' toilet': 'Toilet',
                     '24hour checkin': '24 hour checkin',
                     'Accessibleheight bed': 'Accessible height bed',
                     'Accessibleheight toilet':
                     'Accessible height toilet',
                     'Buzzerwireless intercom': 'Buzzer/Wireless intercom',
                     'Familykid friendly': 'Family/kid friendly',
                     'Highresolution computer monitor':
                     'High resolution computer monitor',
                     'Pack n Playtravel crib': 'Pack-n-Play travel crib',
                     'Roomdarkening shades': 'Room darkening shades',
                     'Self checkin': 'Self check-in',
                     'SkiinSkiout': 'Ski-in/Ski-out',
                     'Stepfree shower': 'Step-free shower',
                     'Washer  Dryer': 'Washer/Dryer',
                     'Welllit path to entrance':
                     'Well-lit path to entrance'}
amenities_series, amenities_set = rename_amenities(amenities_series,
                                                   amenities_set,
                                                   amenities_mapping)
```

Now we have a series of lists of cleaned up amenity names


```python
amenities_series
```




    city           
    seattle    0       {amen_stove, amen_free_parking_on_premises, am...
               1       {amen_heating, amen_hangers, amen_hair_dryer, ...
               2       {amen_stove, amen_heating, amen_first_aid_kit,...
               3       {amen_well-lit_path_to_entrance, amen_heating,...
               4       {amen_stove, amen_heating, amen_wheelchair_acc...
                                             ...                        
    vancouver  6130    {amen_stove, amen_heating, amen_dishwasher, am...
               6145    {amen_stove, amen_heating, amen_bbq_grill, ame...
               6150    {amen_stove, amen_heating, amen_hot_tub, amen_...
               6152    {amen_stove, amen_heating, amen_indoor_firepla...
               6174    {amen_stove, amen_heating, amen_first_aid_kit,...
    Name: amenities, Length: 15983, dtype: object



And a set of all unique amenities. Let's see how many there are


```python
# count number of unique amenities
len(amenities_set)
```




    191



Let's whittle this number down by taking amenities which are don't have extremely unbalanced distributions. We'll find amenities present in more than 10% but less than 90% of listings.


```python
def count_amenities(amenities_series, amenities_set):
    """Count total occurences of each amenity in dataset."""
    amenities_count = {amen: 0 for amen in amenities_set}

    for amens_entry in amenities_series:
        for amen in amens_entry:
            amenities_count[amen] += 1

    return amenities_count

def get_amenities_cols(amenities_series, amenities_count,
                       prop_low=0.1, prop_hi=0.9):
    """Return amenities with proportion in between thresholds."""
    # dataframe of amenities counts
    n = len(amenities_series)
    amenities_prop_df = pd.DataFrame(amenities_count, index=['prop']).T
    amenities_prop_df = amenities_prop_df.sort_values(by='prop')/n
    amenities_prop_df = amenities_prop_df.query('prop >= ' + str(prop_low))
    amenities_prop_df = amenities_prop_df.query('prop <= ' + str(prop_hi))

    return set(amenities_prop_df.index)
```


```python
# get amenities present in > 10% and less than 90% of listings
amenities_count = count_amenities(amenities_series, amenities_set)
amenities_cols = get_amenities_cols(amenities_series, amenities_count,
                                    prop_low=0.1, prop_hi=0.9)
# check number of these
len(amenities_cols)
```




    49



That's a more reasonable number of amenities for us to consider. Finally we'll create a dataframe of boolean variables the presence/absence of these amenities for each listing. We'll also include a variable which counts the number of these amenities present for each listing 


```python
def get_amenities_df(amenities_series, amenities_cols):
    """Create dataframe of amenities variables."""
    amenities_df = pd.DataFrame(columns=amenities_cols,
                                index=amenities_series.index).astype('bool')

    def has_amenity(amenity, amens_entry):
        return amenity in amens_entry

    for amenity in amenities_cols:
        applyfunc = partial(has_amenity, amenity)
        amenities_df.loc[:, amenity] = amenities_series.apply(applyfunc)

    def num_amenities(amenities_cols, amen_entry):
        return len(set(amen_entry).intersection(amenities_cols))

    applyfunc = partial(num_amenities, amenities_cols)
    amenities_df['num_amenities'] = amenities_series.apply(applyfunc)

    return amenities_df
```


```python
# dataframe with amenities as boolean columns
amenities_df = get_amenities_df(amenities_series, amenities_cols)
amenities_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>amen_stove</th>
      <th>amen_free_parking_on_premises</th>
      <th>amen_bathtub</th>
      <th>amen_bbq_grill</th>
      <th>amen_indoor_fireplace</th>
      <th>amen_dishwasher</th>
      <th>amen_first_aid_kit</th>
      <th>amen_hot_water</th>
      <th>amen_hair_dryer</th>
      <th>amen_microwave</th>
      <th>amen_pets_allowed</th>
      <th>amen_coffee_maker</th>
      <th>amen_dryer</th>
      <th>amen_oven</th>
      <th>amen_free_street_parking</th>
      <th>amen_garden_or_backyard</th>
      <th>amen_tv</th>
      <th>amen_cooking_basics</th>
      <th>amen_air_conditioning</th>
      <th>amen_safety_card</th>
      <th>amen_cable_tv</th>
      <th>amen_internet</th>
      <th>amen_luggage_dropoff_allowed</th>
      <th>amen_carbon_monoxide_detector</th>
      <th>amen_keypad</th>
      <th>amen_childrens_books_and_toys</th>
      <th>amen_pack-n-play_travel_crib</th>
      <th>amen_lockbox</th>
      <th>amen_fire_extinguisher</th>
      <th>amen_private_living_room</th>
      <th>amen_dishes_and_silverware</th>
      <th>amen_gym</th>
      <th>amen_bed_linens</th>
      <th>amen_elevator</th>
      <th>amen_kitchen</th>
      <th>amen_family/kid_friendly</th>
      <th>amen_extra_pillows_and_blankets</th>
      <th>amen_washer</th>
      <th>amen_single_level_home</th>
      <th>amen_lock_on_bedroom_door</th>
      <th>amen_iron</th>
      <th>amen_refrigerator</th>
      <th>amen_self_check-in</th>
      <th>amen_pets_live_on_this_property</th>
      <th>amen_long_term_stays_allowed</th>
      <th>amen_patio_or_balcony</th>
      <th>amen_laptop_friendly_workspace</th>
      <th>amen_24_hour_checkin</th>
      <th>amen_private_entrance</th>
      <th>num_amenities</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">seattle</th>
      <th>0</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>35</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">vancouver</th>
      <th>6130</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>24</td>
    </tr>
    <tr>
      <th>6145</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>17</td>
    </tr>
    <tr>
      <th>6150</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>28</td>
    </tr>
    <tr>
      <th>6152</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>25</td>
    </tr>
    <tr>
      <th>6174</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
<p>15983 rows × 50 columns</p>
</div>



And finally, we'll add this to our listings dataframe


```python
# add amenities to listings df
listings_df = pd.concat([listings_df, amenities_df], axis=1)\
              .drop(columns=['amenities'])
```

### Currency conversion

The Vancouver prices are expressed in CAD, so we'll convert to USD. The exchange rate as of November 26, 2019 was 1 CAD = 0.75341 USD.


```python
def conv_cad_to_usd(entry):
    """Currency conversion helper."""
    return round(0.76124 * entry, 0)

def conv_curr_cols(df, curr_cols, curr_conv_func):
    """Convert currency columns."""
    df_cp = df.copy()
    df_cp.loc[:, curr_cols] = \
        df_cp[curr_cols].apply(lambda x: conv_cad_to_usd(x) 
                               if 'vancouver' in x.name else x, axis=1)
    return df_cp
```


```python
# convert currency columns in listings df
list_curr_cols = ['cleaning_fee', 'price', 'security_deposit']
listings_df = conv_curr_cols(listings_df, list_curr_cols, conv_cad_to_usd)
```


```python
# workaround for converting calendar columns due to slowness of conv_curr_cols
cal_van_price = calendar_df.loc['vancouver']['price'].apply(conv_cad_to_usd).values
cal_other_price = calendar_df.loc[['seattle', 'portland']]['price'].values
calendar_df.loc[:, 'price'] = np.append(cal_other_price, cal_van_price)
```

### Save cleaned datasets


```python
# save dfs as h5 files to save write time
listings_df = alphabetize_cols(listings_df)
listings_df.to_hdf('data/listings.h5', key='listings', mode='w', format='table')
calendar_df = alphabetize_cols(calendar_df, first_col='listing_id')
calendar_df.to_hdf('data/calendar.h5', key='calendar', mode='w', format='table')
```


```python
# delete unwanted dataset
datadir = './data'
for item in os.listdir(datadir):
    if item.endswith('csv'):
        os.remove(os.path.join(datadir, item))
```
