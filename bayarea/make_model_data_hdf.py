
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import subprocess
from subprocess import PIPE


# In[2]:


d = './data/'


# In[8]:


parcels = pd.read_csv(
        d + 'parcel_attr.csv',
        index_col='primary_id',
        dtype={'primary_id': int, 'block_id': str})


# In[9]:


parcels.head()


# In[10]:


buildings = pd.read_csv(
        d + 'buildings_v2.csv',
        index_col='building_id', dtype={'building_id': int, 'parcel_id': int})
buildings['res_sqft_per_unit'] = buildings['residential_sqft'] / buildings['residential_units']
buildings['res_sqft_per_unit'][buildings['res_sqft_per_unit'] == np.inf] = 0


# In[11]:


building_types = pd.read_csv(
    d + 'building_types.csv',
    index_col='building_type_id', dtype={'building_type_id': int})


# In[12]:


building_types.head()


# In[13]:


rentals = pd.read_csv(
        d + 'MTC_craigslist_listings_7-10-18.csv',
        index_col='pid', dtype={
            'pid': int, 'date': str, 'region': str, 'neighborhood': str,
            'rent': float, 'sqft': float, 'rent_sqft': float, 
            'longitude': float, 'latitude': float, 'county': str,
            'fips_block': str, 'state': str, 'bathrooms': str})


# In[14]:


units = pd.read_csv(
        d + 'units_v2.csv',
        index_col='unit_id', dtype={'unit_id': int, 'building_id': int})


# In[15]:


households = pd.read_csv(
        d + 'households_v2.csv',
        index_col='household_id', dtype={
            'household_id': int, 'block_group_id': str, 'state': str,
            'county': str, 'tract': str, 'block_group': str,
            'building_id': int, 'unit_id': int, 'persons': float})


# In[16]:


persons = pd.read_csv(
        d + 'persons_v3.csv',
        index_col='person_id', dtype={'person_id': int, 'household_id': int})


# In[17]:


jobs = pd.read_csv(
        d + 'jobs_v2.csv',
        index_col='job_id', dtype={'job_id': int, 'building_id': int})


# In[18]:


establishments = pd.read_csv(
        d + 'establishments_v2.csv',
        index_col='establishment_id', dtype={
            'establishment_id': int, 'building_id': int, 'primary_id': int})


# In[19]:


b = 'data/beam_to_urbansim-v3/'
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


# In[20]:


nodesbeam = pd.read_csv(b + beam_nodes_fname).set_index('id')


# In[21]:


edgesbeam = pd.read_csv(b + beam_links_filtered_fname).set_index('link')


# In[22]:


nodesbeam.head()


# In[23]:


edgesbeam.head()


# In[24]:


nodeswalk = pd.read_csv(d + 'bayarea_walk_nodes.csv').set_index('osmid')
edgeswalk = pd.read_csv(d + 'bayarea_walk_edges.csv')


# In[25]:


nodeswalk.head()


# In[26]:


edgeswalk.head()


# In[27]:


nodessmall = pd.read_csv(d + 'bay_area_tertiary_strongly_nodes.csv').set_index('osmid')
edgessmall = pd.read_csv(d + 'bay_area_tertiary_strongly_edges.csv')


# In[28]:


nodessmall.head()


# In[29]:


store = pd.HDFStore('data/model_data.h5')
store.put('parcels',parcels)
store.put('buildings',buildings)
store.put('building_types',building_types)
store.put('units',units)
store.put('rentals',rentals)
store.put('households',households)
store.put('persons',persons)
store.put('jobs',jobs)
store.put('establishments',establishments)
store.put('nodesbeam',nodesbeam)
store.put('edgesbeam',edgesbeam)
store.put('nodeswalk',nodeswalk)
store.put('edgeswalk',edgeswalk)
store.put('nodessmall',nodeswalk)
store.put('edgessmall',edgessmall)
store.keys()


# In[30]:


store.close()

