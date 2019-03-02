import orca
import numpy as np
import pandas as pd


# data documentation: https://berkeley.app.box.com/notes/282712547032


# Set data directory

d = './data/'

if 'data_directory' in orca.list_injectables():
    d = orca.get_injectable('data_directory')


@orca.injectable('store', cache=True)
def hdfstore():
    return pd.HDFStore(
        os.path.join(misc.data_dir(), "model_data.h5"),
        mode='r')


@orca.table('parcels', cache=True)
def parcels(store):
    df = store['parcels']
    df.index.name = 'parcel_id'
    return df


@orca.table('buildings', cache=True)
def buildings(store):
    df = store['buildings']
    return df


@orca.table('jobs', cache=True)
def jobs(store):
    df = store['jobs']
    return df


@orca.table('establishments', cache=True)
def establishments(store):
    df = store['establishments']
    return df


@orca.table('households', cache=True)
def households(store):
    df = store['households']
    return df


@orca.table('persons', cache=True)
def persons(store):
    df = store['persons']
    return df


@orca.table('craigslist', cache=True)
def craigslist(store):
    craigslist = store['craigslist']
    craigslist.rent[craigslist.rent < 100] = 100
    craigslist.rent[craigslist.rent > 10000] = 10000

    craigslist.rent_sqft[craigslist.rent_sqft < .2] = .2
    craigslist.rent_sqft[craigslist.rent_sqft > 50] = 50
    return craigslist


@orca.table('units', cache=True)
def units(store):
    df = store['units']
    df.index.name = 'unit_id'
    return df


@orca.table('nodessmall', cache=True)
def nodessmall(store):
    df = store['nodessmall']
    df.index.name = 'osmid'
    return df


@orca.table('edgessmall', cache=True)
def edgessmall(store):
    df = store['edgessmall']
    return df


@orca.table('nodeswalk', cache=True)
def nodeswalk(store):
    df = store['nodeswalk']
    df.index.name = 'osmid'
    return df


@orca.table('edgeswalk', cache=True)
def edgessmall(store):
    df = store['edgeswalk']
    return df


@orca.table('nodesbeam', cache=True)
def nodesbeam(store):
    df = store['nodesbeam']
    df.index.name = 'id'
    return df


@orca.table('edgesbeam', cache=True)
def edgesbeam(store):
    df = store['edgesbeam']
    return df


# Broadcasts, a.k.a. merge relationships


orca.broadcast(
    'parcels', 'buildings', cast_index=True, onto_on='parcel_id')
orca.broadcast(
    'buildings', 'units', cast_index=True, onto_on='building_id')
orca.broadcast(
    'units', 'households', cast_index=True, onto_on='unit_id')
orca.broadcast(
    'households', 'persons', cast_index=True, onto_on='household_id')
orca.broadcast(
    'buildings', 'jobs', cast_index=True, onto_on='building_id')
orca.broadcast(
    'buildings', 'establishments', cast_index=True, onto_on='building_id')
orca.broadcast(
    'nodeswalk', 'parcels', cast_index=True, onto_on='node_id_walk')
orca.broadcast(
    'nodeswalk', 'craigslist', cast_index=True, onto_on='node_id_walk')
orca.broadcast(
    'nodessmall', 'craigslist', cast_index=True, onto_on='node_id_small')
orca.broadcast(
    'nodessmall', 'parcels', cast_index=True, onto_on='node_id_small')
orca.broadcast(
    'nodesbeam', 'parcels', cast_index=True, onto_on='node_id_beam')
orca.broadcast(
    'nodesbeam', 'craigslist', cast_index=True, onto_on='node_id_beam')
