import pandas as pd
import numpy as np
import subprocess
from subprocess import PIPE


def send_df_to_s3_as_pq(df, table_name):
    s3_url = 's3://urbansim-baseyear-inputs/{0}.parquet.gz'.format(table_name)
    df.to_parquet(s3_url, compression='gzip', engine='pyarrow')


d = '/home/data/fall_2018/'
b = '/home/data/spring_2019/beam_to_urbansim-v3/'

parcels = pd.read_csv(
    d + 'parcel_attr.csv',
    index_col='primary_id',
    dtype={'primary_id': int, 'block_id': str})

buildings = pd.read_csv(
    d + 'buildings_v2.csv',
    index_col='building_id', dtype={'building_id': int, 'parcel_id': int})
buildings['res_sqft_per_unit'] = buildings['residential_sqft'] / buildings[
    'residential_units']
buildings['res_sqft_per_unit'][buildings['res_sqft_per_unit'] == np.inf] = 0

building_types = pd.read_csv(
    d + 'building_types.csv',
    index_col='building_type_id', dtype={'building_type_id': int})

rentals = pd.read_csv(
    d + 'MTC_craigslist_listings_7-10-18.csv',
    index_col='pid', dtype={
        'pid': int, 'date': str, 'region': str, 'neighborhood': str,
        'rent': float, 'sqft': float, 'rent_sqft': float,
        'longitude': float, 'latitude': float, 'county': str,
        'fips_block': str, 'state': str, 'bathrooms': str})

units = pd.read_csv(
    d + 'units_v2.csv',
    index_col='unit_id', dtype={'unit_id': int, 'building_id': int})

households = pd.read_csv(
    d + 'households_v2.csv',
    index_col='household_id', dtype={
        'household_id': int, 'block_group_id': str, 'state': str,
        'county': str, 'tract': str, 'block_group': str,
        'building_id': int, 'unit_id': int, 'persons': float})

persons = pd.read_csv(
    d + 'persons_v3.csv',
    index_col='person_id', dtype={'person_id': int, 'household_id': int})

jobs = pd.read_csv(
    d + 'jobs_v2.csv',
    index_col='job_id', dtype={'job_id': int, 'building_id': int})

establishments = pd.read_csv(
    d + 'establishments_v2.csv',
    index_col='establishment_id', dtype={
        'establishment_id': int, 'building_id': int, 'primary_id': int})

beam_nodes_fname = 'beam-network-nodes.csv'
beam_links_fname = '10.linkstats.csv'
beam_links_filtered_fname = 'beam_links_8am.csv'
with open(b + beam_links_filtered_fname, 'w') as f:
    p1 = subprocess.Popen(
        ["cat", b + beam_links_fname], stdout=PIPE)
    p2 = subprocess.Popen([
        "awk", "-F", ",",
        '(NR==1) || ($4 == "8.0" && $8 == "AVG")'],
        stdin=p1.stdout, stdout=f)
    p2.wait()

nodesbeam = pd.read_csv(b + beam_nodes_fname).set_index('id')
edgesbeam = pd.read_csv(b + beam_links_filtered_fname).set_index('link')

nodeswalk = pd.read_csv(
    d + 'bayarea_walk_nodes.csv', low_memory=False).set_index('osmid')
edgeswalk = pd.read_csv(
    d + 'bayarea_walk_edges.csv', low_memory=False).set_index('uniqueid')

nodessmall = pd.read_csv(
    d + 'bay_area_tertiary_strongly_nodes.csv').set_index('osmid')
edgessmall = pd.read_csv(
    d + 'bay_area_tertiary_strongly_edges.csv').set_index('uniqueid')

skims = pd.read_csv('./data/skims_110118.csv', index_col=0)

for table_str in [
        'parcels', 'buildings', 'building_types', 'units', 'rentals',
        'households', 'persons', 'jobs', 'establishments', 'nodesbeam',
        'edgesbeam', 'nodeswalk', 'edgeswalk', 'nodessmall', 'edgessmall',
        'skims']:
    print(table_str)
    send_df_to_s3_as_pq(eval(table_str), table_str)
