"""Wrangle datasets and store results."""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from pandas.api.types import CategoricalDtype
from functools import partial


def load_dfs(file_prefixes, city_names, files_path):
    """Dataframes for all but .geojson data files for each city."""
    dfs = defaultdict(dict)

    for file_prefix in file_prefixes:
        for city_name in city_names:
            file_path = files_path + city_name + '_' + file_prefix + '.csv'
            dfs[file_prefix][city_name] = pd.read_csv(file_path)

    return dfs


def merge_dfs(dfs, file_prefixes, city_names):
    """Merge dataframes for each kind of data file."""
    merged_dfs = defaultdict(list)

    for file_prefix in file_prefixes:
        dfs_list = [dfs[file_prefix][city_name] for city_name in city_names]
        merged_dfs[file_prefix] = pd.concat(dfs_list, keys=city_names,
                                            names=['city'])
        try:
            merged_dfs[file_prefix].drop(columns=['city'], inplace=True)
        except KeyError:
            pass

    return merged_dfs


def prop_missing_vals(df, axis=0, threshold=0.25):
    """Proportions missing values greater than threshold."""
    prop = df.isna().sum(axis=axis).sort_values()/df.shape[axis]
    return prop[prop > threshold]


def prop_missing_vals_df(df, axis=0, threshold=0.25, ascending=False):
    """Get missing values in df by proportion above threshold along axis."""
    prop = df.isna().sum(axis=axis).sort_values(ascending=ascending)
    prop = prop/df.shape[axis]
    prop.name = 'prop_miss_vals'
    return pd.DataFrame(prop[prop > threshold])


def alphabetize_cols(df, first_col='id'):
    """Alphabetize columns."""
    df_cp = df.copy()
    alph_cols = list(df_cp.columns.sort_values())
    alph_cols.remove(first_col)
    alph_cols = [first_col] + alph_cols
    df_cp = df_cp[alph_cols]
    return df_cp


def can_conv_to_int(float_vars):
    """Check which float values can be converted to int without rounding."""
    res = (np.abs(np.floor(float_vars) - float_vars) > 0).any()
    return ~ res


def set_ord_cat_dtypes(listings_df, conv_dtypes, ord_cat_cols):
    """Set ordered categorical columns dtypes."""
    listings_df_cp = listings_df.copy()
    for (col, ordering) in conv_dtypes['ordered_categorical'].items():
        dtype = CategoricalDtype(categories=ordering, ordered=True)
        listings_df_cp[col] = listings_df_cp[col].astype(dtype)
    return listings_df_cp


def conv_cad_to_usd(entry):
    """Currency conversion helper."""
    return 0.75341*entry


def conv_to_float(entry):
    """Float conversion helper."""
    try:
        return entry.replace('$', '').replace('%', '').replace(',', '')
    except AttributeError:
        return entry


def conv_to_bool(entry):
    """Booleans conversion helper."""
    return entry == 't'


def conv_cols(listings_df, conv_dtypes, conv_func, dtype_name):
    """Process and convert some columns to float dtype."""
    listings_df_cp = listings_df.copy()

    for col in conv_dtypes[dtype_name]:
        listings_df_cp[col] = listings_df[col].apply(conv_func)\
                              .astype(dtype_name)

    return listings_df_cp


def conv_curr_cols(df, curr_cols, curr_conv_func):
    """Convert currency columns."""
    df_cp = df.copy()
    df_cp.loc[:, curr_cols] = \
        df_cp[curr_cols].apply(lambda x: conv_cad_to_usd(x) 
                               if 'vancouver' in x.name else x, axis=1)
    return df_cp


def backfill_missing_prices(calendar_df, two_listings_df):
    """Backfill missing price data for two listings in calendar dataset."""
    week_delta = pd.Timedelta(1, unit='w')
    calendar_df_cp = calendar_df.copy()
    for index in two_listings_df.index:
        listing_id = two_listings_df.loc[index]['listing_id']
        one_week_ago = two_listings_df.loc[index]['date'] - week_delta
        mask = ((calendar_df_cp['listing_id'] == listing_id) &
                (calendar_df_cp['date'] == one_week_ago))
        price = calendar_df_cp[mask]['price'].values[0]
        calendar_df_cp.loc[index, 'price'] = price
    return calendar_df_cp


def process_amenities(amenities_series):
    """Process entries in amenities series."""
    # convert amenities lists into sets of strings
    amenities_series = amenities_series.apply(lambda x: set(x.split(',')))

    # set for tracking unique amenties
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


if __name__ == '__main__':

    #
    # load dfs
    #

    file_prefixes = ['listings', 'calendar']
    city_names = ['seattle', 'portland', 'vancouver']
    files_path = 'data/'
    dfs = load_dfs(file_prefixes, city_names, files_path)
    dfs = merge_dfs(dfs, file_prefixes, city_names)
    listings_df = dfs['listings']
    calendar_df = dfs['calendar']

    #
    # drop rows/columns from listings dataset
    #

    drop_cols = set(prop_missing_vals_df(listings_df, axis=0).index)
    more_drop_cols = {'description', 'house_rules', 'name', 'space',
                      'summary', 'calendar_updated', 'host_location',
                      'host_name', 'host_neighbourhood', 'host_picture_url',
                      'host_thumbnail_url', 'host_url', 'host_verifications',
                      'is_location_exact', 'jurisdiction_names',
                      'listing_url', 'market', 'name', 'picture_url',
                      'scrape_id', 'maximum_minimum_nights',
                      'minimum_maximum_nights', 'maximum_maximum_nights',
                      'minimum_minimum_nights', 'minimum_nights_avg_ntm',
                      'maximum_nights_avg_ntm', 'last_scraped',
                      'calendar_last_scraped', 'country', 'country_code',
                      'experiences_offered', 'has_availability',
                      'host_has_profile_pic', 'smart_location', 'street',
                      'state', 'host_listings_count',
                      'host_total_listings_count', 'neighbourhood',
                      'zipcode', 'number_of_reviews_ltm'}
    drop_cols = drop_cols.union(more_drop_cols)
    listings_df = listings_df.drop(columns=drop_cols)
    listings_df = alphabetize_cols(listings_df)

    #
    # enforce dtypes for listings dataset
    #

    # dict for conversion type
    conv_dtypes = defaultdict(set)
    # int variables
    conv_to_int = can_conv_to_int(listings_df.select_dtypes('float64')
                                  .dropna())
    conv_to_int_cols = set(conv_to_int[conv_to_int.T].index)
    conv_dtypes['int'] = conv_to_int_cols
    # non-ordered categorical variables
    conv_dtypes['categorical'] = {'host_identity_verified',
                                  'host_is_superhost', 'instant_bookable',
                                  'is_business_travel_ready',
                                  'neighbourhood_cleansed', 'property_type',
                                  'require_guest_phone_verification',
                                  'require_guest_profile_picture',
                                  'requires_license'}
    conv_dtypes['datetime'] = {'first_review', 'host_since', 'last_review'}
    listings_df.loc[:, conv_dtypes['categorical']] = \
        listings_df[conv_dtypes['categorical']].astype('category')
    listings_df.loc[:, conv_dtypes['datetime']] = \
        listings_df[conv_dtypes['datetime']].astype('datetime64')
    # ordered categorical variables
    ord_cat_cols = ['bed_type', 'cancellation_policy', 'host_response_time',
                    'room_type']
    conv_dtypes['ordered_categorical'] = {col: None for col in ord_cat_cols}
    conv_dtypes['ordered_categorical']['bed_type'] = ['Couch', 'Airbed',
                                                      'Pull-out Sofa',
                                                      'Futon', 'Real Bed']
    conv_dtypes['ordered_categorical']['cancellation_policy'] = \
                                      ['super_strict_60', 'super_strict_30',
                                       'strict',
                                       'strict_14_with_grace_period',
                                       'moderate', 'flexible']
    conv_dtypes['ordered_categorical']['host_response_time'] = \
                                      ['within an hour', 'within a few hour',
                                       'within a day', 'a few days of more']
    conv_dtypes['ordered_categorical']['room_type'] = \
                                      ['Shared room', 'Hotel room',
                                       'Private room', 'Entire home/apt']
    listings_df_cp = set_ord_cat_dtypes(listings_df, conv_dtypes,
                                        ord_cat_cols)
    # some float variables
    conv_dtypes['float'] = {'cleaning_fee', 'extra_people', 'price',
                            'security_deposit', 'host_response_rate'}
    listings_df = conv_cols(listings_df, conv_dtypes, conv_to_float, 'float')

    #
    # deal with missing values in listings dataset
    #

    # drop rows missing any review variables
    miss_vals_df = prop_missing_vals_df(listings_df, axis=0, threshold=0)
    miss_vals_df = pd.DataFrame({'col': miss_vals_df.index,
                                 'prop_miss':
                                 miss_vals_df['prop_miss_vals'].values,
                                 'dtype':
                                 listings_df[miss_vals_df.index].dtypes})

    miss_val_rev_cols = [col for col in miss_vals_df.index if 'review'
                         in col]
    mask = listings_df[miss_val_rev_cols].isna().sum(axis=1) == 0
    listings_df = listings_df[mask]
    # drop two more review columns
    listings_df = listings_df.drop(columns=['first_review', 'last_review'])
    # dictionary for imputation values
    impute_vals = defaultdict(None)
    # imputation values by column
    for col in ['bathrooms', 'beds', 'bedrooms', 'host_since',
                'host_is_superhost', 'host_identity_verified']:
        impute_vals[col] = listings_df[col].mode().values[0]
    impute_vals['security_deposit'] = 0
    impute_vals['cleaning_fee'] = \
        listings_df['cleaning_fee'].dropna().median()
    for col in ['host_response_rate', 'host_response_time']:
        impute_vals[col] = listings_df[col].mode().values[0]
    # impute all missing values
    listings_df = listings_df.fillna(impute_vals)

    #
    # downcast dtypes for listings dataset
    #

    listings_df.loc[:, conv_dtypes['int']] = \
        listings_df[conv_dtypes['int']].astype('int')
    conv_dtypes['bool'] = {'host_identity_verified',
                           'host_is_superhost', 'instant_bookable',
                           'is_business_travel_ready',
                           'require_guest_phone_verification',
                           'require_guest_profile_picture',
                           'requires_license'}
    listings_df = conv_cols(listings_df, conv_dtypes, conv_to_bool, 'bool')

    #
    # drop rows/cols in calendar dataset
    #

    miss_vals_df = calendar_df.loc[calendar_df.isna().sum(axis=1) > 0]
    miss_cal_idx = miss_vals_df.reset_index(level=0)\
                   .groupby(['city', 'listing_id'])['listing_id'].count()
    listing_ids = {index[1] for index in
                   miss_cal_idx[miss_cal_idx > 7].index}
    list_ids_mask = ~ listings_df['id'].apply(lambda x: x in listing_ids)
    listings_df = listings_df[list_ids_mask]
    cal_ids_mask = ~ calendar_df['listing_id']\
                     .apply(lambda x: x in listing_ids)
    calendar_df = calendar_df[cal_ids_mask]
    calendar_df = calendar_df.drop(columns=['adjusted_price'])

    #
    # enforce dtypes in calendar dataset
    #

    calendar_df.loc[:, 'date'] = calendar_df['date'].astype('datetime64')
    conv_dtypes = defaultdict(set)
    conv_dtypes['bool'] = {'available'}
    conv_dtypes['float'] = {'price'}
    calendar_df = conv_cols(calendar_df, conv_dtypes, conv_to_bool, 'bool')
    calendar_df = conv_cols(calendar_df, conv_dtypes, conv_to_float, 'float')

    #
    # deal with missing values in calendar dataset
    #

    listing_ids = [index[1] for index
                   in miss_cal_idx[miss_cal_idx <= 7].index]
    cal_id_mask = (calendar_df['listing_id'].
                   apply(lambda x: x in listing_ids) &
                   (calendar_df['price'].isna()))
    two_listings_df = calendar_df[cal_id_mask]
    calendar_df = backfill_missing_prices(calendar_df, two_listings_df)

    #
    # synchronize date column in calendar dataset
    #

    min_date = calendar_df.loc['seattle']['date'].min()
    max_date = calendar_df.loc['vancouver']['date'].max()
    cal_date_mask = ((calendar_df['date'] >= min_date) &
                     (calendar_df['date'] <= max_date))
    calendar_df = calendar_df[cal_date_mask]
    calendar_df = backfill_missing_prices(calendar_df, two_listings_df)

    #
    # create amenities features
    #

    amenities_series, amenities_set = process_amenities(
                                      listings_df['amenities'])
    amenities_mapping = {' toilet': 'Toilet',
                         '24hour checkin': '24 hour checkin',
                         'Accessibleheight bed': 'Accessible height bed',
                         'Accessibleheight toilet':
                         'Accessible height toilet',
                         'Buzzerwireless intercom':
                         'Buzzer/Wireless intercom',
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

    amenities_count = count_amenities(amenities_series, amenities_set)
    amenities_cols = get_amenities_cols(amenities_series, amenities_count,
                                        prop_low=0.1, prop_hi=0.9)
    amenities_df = get_amenities_df(amenities_series, amenities_cols)
    listings_df = pd.concat([listings_df, amenities_df], axis=1)\
                  .drop(columns=['amenities'])

    #
    # currency conversions
    #

    list_curr_cols = ['cleaning_fee', 'price', 'security_deposit']
    listings_df = conv_curr_cols(listings_df, list_curr_cols,
                                 conv_cad_to_usd)
    # workaround due to slowness of conv_curr_cols on calendar_df
    cal_van_price = calendar_df.loc['vancouver']['price']\
                    .apply(conv_cad_to_usd).values
    cal_other_price = calendar_df.loc[['seattle', 'portland']]['price']\
                      .values
    calendar_df.loc[:, 'price'] = np.append(cal_other_price, cal_van_price)

    # save datasets
    listings_df = alphabetize_cols(listings_df, first_col='id')
    listings_df.to_hdf('data/listings.h5', key='listings',
                       mode='w', format='table')
    calendar_df = alphabetize_cols(calendar_df, first_col='listing_id')
    calendar_df.to_hdf('data/calendar.h5', key='calendar',
                       mode='w', format='table')
