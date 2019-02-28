import os
import numpy as np
import pandas as pd

import orca
from urbansim.utils import misc


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


@orca.table('households', cache=True)
def households(store):
    df = store['households']
    return df


@orca.table('households_pums', cache=True)
def households_pums(store):
    df = store['pums']
    return df


@orca.table('travel_data', cache=True)
def travel_data(store):
    df = store['travel_data'].reset_index()
    df.columns = ['from_zone_id', 'to_zone_id', 'am_peak_travel_time',
                  'md_offpeak_travel_time']
    df = df.set_index(['from_zone_id', 'to_zone_id'])
    return df


@orca.table('nodes', cache=True)
def nodes(store):
    df = store['nodes']
    df.index.name = 'node_id'
    return df


@orca.table('edges', cache=True)
def edges(store):
    df = store['edges']
    return df


@orca.table('annual_employment_control_totals', cache=True)
def aect(store):
    df = store['annual_employment_control_totals']
    return df


@orca.table('annual_household_control_totals', cache=True)
def ahct(store):
    df = store['annual_household_control_totals']
    return df


@orca.table('craigslist', cache=True)
def craigslist(store):
    craigslist = store['craigslist']
    craigslist.rent[craigslist.rent < 100] = 100
    craigslist.rent[craigslist.rent > 10000] = 10000

    craigslist.rent_sqft[craigslist.rent_sqft < .2] = .2
    craigslist.rent_sqft[craigslist.rent_sqft > 50] = 50
    return craigslist


def register_aggregation_table(table_name, table_id):
    """
    Generator function for tables representing aggregate geography.
    """

    @orca.table(table_name, cache=True)
    def func(parcels):
        geog_ids = parcels[table_id].value_counts().index.values
        df = pd.DataFrame(index=geog_ids)
        df.index.name = table_id
        return df

    return func


aggregate_geos = {'zonings': 'zoning_id',
                  'locations': 'location_id',
                  'block_groups': 'block_group_id',
                  'blocks': 'block_id',
                  'zones': 'zone_id',
                  'plans': 'plan_id',
                  'zone_districts': 'zone_district_id',
                  'zone_subdistricts': 'zone_subdistrict_id'}
orca.add_injectable('aggregate_geos', aggregate_geos)

for geog in aggregate_geos.items():
    register_aggregation_table(geog[0], geog[1])


@orca.injectable('year')
def year():
    default_year = 2010
    try:
        iter_var = orca.get_injectable('iter_var')
        if iter_var is not None:
            return iter_var
        else:
            return default_year
    except:
        return default_year


@orca.table('plan_types', cache=True)
def plan_types():
    df = pd.read_csv('./data/plan_types.csv').set_index('plan_id')
    return df


@orca.table('zone_types', cache=True)
def zone_types():
    df = pd.read_csv('./data/zone_types.csv').set_index('zoning_id')
    return df

  
@orca.table('plan_compatible_zones', cache=True)
def plan_compatible_zones():
    df = pd.read_csv('./data/plan_compatible_zones.csv').\
        set_index('plan_zone_id')
    return df

@orca.table('building_types', cache=True)
def building_types():
    df = pd.read_csv('./data/building_types.csv').set_index('building_type_id')
    return df

@orca.table('allowable_building_types', cache=True)
def allowable_building_types():
    df = pd.read_csv('./data/allowable_building_types.csv').\
        set_index('zoning_building_id')
    return df

  
@orca.table('building_sqft_per_job', cache=True)
def building_sqft_per_job():
    df = pd.read_csv('./data/bsqft_per_job.csv')
    return df


@orca.table('zone_overlay_types', cache=True)
def zone_overlay_types():
    df = pd.read_csv('./data/zone_overlay_types.csv')
    return df


@orca.table('site_proposals', cache=False)
def site_proposals(parcels, zone_types, plan_compatible_zones):
    # Prepares input files
    parcelsdf = parcels.local.reset_index()
    zone_typesdf = zone_types.to_frame().reset_index()
    plan_compatible_zonesdf = plan_compatible_zones.to_frame()

    # Identifies parcel location ("status_ugb")
    parcelsdf = defines_location(parcelsdf)

    # Creates possible parcel_zoning combinations
    site_proposals = creates_site_proposals\
        (parcelsdf, plan_compatible_zonesdf, zone_typesdf)

    #Calculates rezoning costs if applicable
    site_proposals = rezoning_costs(site_proposals)

    # Calculates overlay costs if applicable
    site_proposals = overlay_costs(site_proposals)

    # Formats output
    site_proposals = formats_site_proposals(site_proposals)

    return site_proposals


def defines_location(parcelsdf):
    parcelsdf.loc[parcelsdf['city'].notnull(),'status_ugb'] = 'within_city'
    parcelsdf.loc[(parcelsdf['city'].isnull()) &
                  (parcelsdf['ugb'].notnull()),'status_ugb'] = 'within_ugb'
    parcelsdf.loc[(parcelsdf['city'].isnull()) &
                  (parcelsdf['ugb'].isnull()),'status_ugb'] = 'outside_ugb'
    return parcelsdf


def creates_site_proposals(parcelsdf, plan_compatible_zonesdf, zone_typesdf):
    # parcels without zoning_id are removed from site_proposals
    parcelsdf[['zoning_id', 'plan_id']] = \
        parcelsdf[['zoning_id', 'plan_id']].fillna(value=0)
    parcelsdf = parcelsdf[parcelsdf['zoning_id'] != 0]

    # Identifies valid plan_zoning combinations existing in parcels table but
    # missing in plan_compatible_zones table. This ensures that all existing
    # parcel-zone combinations are also included in site_proposals
    missing_plan_zoning_combinations = \
        missing_plan_zone_comb(parcelsdf, plan_compatible_zonesdf)

    # Merges plan_compatible_zones table to parcels table to create
    # all potential parcel_zoning combinations
    plan_compatible_zonesdf = plan_compatible_zonesdf[
        ['plan_id', 'zoning_id', 'cost_in_city',
         'cost_in_ugb', 'cost_outside_ugb']]
    plan_compatible_zonesdf = plan_compatible_zonesdf.rename(
        columns={'zoning_id': 'potential_zoning_id',
                 'cost_in_city': 'cost_in_city_',
                 'cost_in_ugb': 'cost_in_ugb_',
                 'cost_outside_ugb': 'cost_outside_ugb_'})

    site_proposals = pd.merge(
        parcelsdf, plan_compatible_zonesdf, on='plan_id', how='left')

    # Parcels that have zoning_id information but no plan_id information
    # are only represented with original zoning_id
    site_proposals.loc[(site_proposals.plan_id == 0) &
                       (site_proposals.zoning_id != 0),
                       'potential_zoning_id'] = site_proposals['zoning_id']

    # Parcels that have a plan_id that doesn't exist in the
    # plan_compatible_zones table and Plans with zoning_id = 0 in the
    # plan_compatible_zones table can be identified with null and zero
    # 'potential_zoning_id`, respectively. This variable is filled with
    # `zoning_id`  in these cases, to represent the original zoning_id only
    site_proposals.loc[site_proposals.potential_zoning_id.isnull(),
                       'potential_zoning_id'] = site_proposals['zoning_id']
    site_proposals.loc[site_proposals.potential_zoning_id == 0,
                       'potential_zoning_id'] = site_proposals['zoning_id']

    # Appends missing plan_zoning combinations to the site_proposals table
    site_proposals = \
        site_proposals.append(missing_plan_zoning_combinations).reset_index()
    site_proposals.loc[site_proposals.missing == 1, 'potential_zoning_id'] = \
        site_proposals['zoning_id']
    site_proposals.drop(columns=['missing'], inplace = True)

    # Removes site proposals that would require rezoning but have
    # can_rezone==True
    zone_typesdf = \
        zone_typesdf.rename(columns={'zoning_id': 'potential_zoning_id'})
    site_proposals = pd.merge(
        site_proposals, zone_typesdf, on = 'potential_zoning_id', how = 'left')
    site_proposals['remove'] = 0
    site_proposals.loc[(site_proposals['zoning_id']!=
                        site_proposals['potential_zoning_id']) &
                       (site_proposals['can_rezone']==0), 'remove'] = 1
    site_proposals = site_proposals[site_proposals['remove'] == 0]

    return site_proposals


def missing_plan_zone_comb(parcelsdf, plan_compatible_zonesdf):
    possible = plan_compatible_zonesdf[['plan_id', 'zoning_id']].copy()
    possible = possible[possible['plan_id'] != 0]
    possible = possible[possible['zoning_id'] != 0]
    possible['represented'] = 1
    actual = parcelsdf[parcelsdf['plan_id'] != 0].copy()
    actual = actual.merge(possible, on=['plan_id', 'zoning_id'], how='left')
    missing = actual[(actual['represented'] != 1)].copy()
    missing = missing[missing['zoning_id'] != 0]
    missings = missing[missing['plan_id'] != 0]
    missing = missing.drop(columns=['represented']).copy()
    missing['potential_zoning_id'] = missing['zoning_id']
    missing['cost_in_city_'] = 0
    missing['cost_in_ugb_'] = 0
    missing['cost_outside_ugb_'] = 0
    missing['missing'] = 1
    return missing

def rezoning_costs(site_proposals):
    # Identifies combinations that imply rezoning
    site_proposals.loc[site_proposals.zoning_id !=
                       site_proposals.potential_zoning_id, 'rezoning'] = 1
    site_proposals.loc[site_proposals['rezoning'] != 1, 'rezoning_cost'] = 0

    # Includes column with rezoning_cost (considering status_ugb)
    site_proposals.loc[(site_proposals['rezoning'] == 1) &
                       (site_proposals['status_ugb'] == 'within_city'),
                       'rezoning_cost'] = site_proposals['cost_in_city_']
    site_proposals.loc[(site_proposals['rezoning'] == 1) &
                       (site_proposals['status_ugb'] == 'within_ugb'),
                       'rezoning_cost'] = site_proposals['cost_in_ugb_']
    site_proposals.loc[
        (site_proposals['rezoning'] == 1) &
        (site_proposals['status_ugb'] == 'outside_ugb'), 'rezoning_cost'] = \
        site_proposals['cost_outside_ugb_']
    site_proposals = \
        site_proposals.drop(columns=['cost_in_city_', 'cost_in_ugb_',
                                     'cost_outside_ugb_', 'rezoning'])
    return site_proposals


def overlay_costs(site_proposals):

    # Includes column with overlay_cost
    # (considering location in relation to ugb)
    overlays = orca.get_table('zone_overlay_types').to_frame()
    overlays = overlays[['overlay_id', 'annexed_overlay_id',
                         'overlay_combination' , 'cost_in_city', 'cost_in_ugb',
                         'cost_outside_ugb']].copy()
    overlays =  overlays.rename(columns={'cost_in_city': 'cost_in_city_',
                                         'cost_in_ugb': 'cost_in_ugb_',
                                         'cost_outside_ugb':
                                             'cost_outside_ugb_'})

    site_proposals.loc[site_proposals.overlay_id.isnull(), 'overlay_id'] = '-1'
    site_proposals['overlay_id'] = \
        site_proposals['overlay_id'].astype(float).astype(int)
    site_proposals = \
        pd.merge(site_proposals, overlays, on='overlay_id', how = 'left')
    site_proposals.loc[site_proposals['status_ugb'] == 'within_city',
                       'overlay_cost'] = site_proposals['cost_in_city_']
    site_proposals.loc[site_proposals['status_ugb'] == 'within_ugb',
                       'overlay_cost'] = site_proposals['cost_in_ugb_']
    site_proposals.loc[site_proposals['status_ugb'] == 'outside_ugb',
                       'overlay_cost'] = site_proposals['cost_outside_ugb_']
    site_proposals = site_proposals.drop\
        (columns=['cost_in_city_', 'cost_in_ugb_', 'cost_outside_ugb_'])

    return site_proposals

def formats_site_proposals(site_proposals):
    # Removes irrelevant fields and renames "potential_zoning_id" to
    # "parcel_zoning_id_combination", unique to each combination in the table
    site_proposals['parcel_zoning_id_combination'] = \
        site_proposals['parcel_id'].astype(int).astype(str) + "_" + \
        site_proposals['potential_zoning_id'].astype(int).astype(str)
    site_proposals = site_proposals.rename\
        (columns={'zoning_id': "original_zoning_id"})

    # Reorders columns to have newly created columns at the beggining.
    ordered_columns = ['parcel_zoning_id_combination', 'parcel_id',
                       'primary_id', 'zone_id','x', 'y','block_group_id',
                       'block_id', 'zone_district_id','zone_subdistrict_id',
                       'location_id','city', 'ugb','status_ugb','plan_id',
                       'overlay_id', 'annexed_overlay_id','original_zoning_id',
                       'zoning_name','potential_zoning_id','can_rezone',
                       'rezoning_cost', 'overlay_cost', 'land_value', 'acres',
                       'proportion_undevelopable','Shape_Length', 'Shape_Area',
                       'max_far','placeholder_max_far', 'max_dua',
                       'placeholder_max_dua','min_far', 'min_dua',
                       'max_height', 'min_front_setback','max_front_setback',
                       'rear_setback','side_setback','coverage', 'OBJECTID']


    site_proposals = site_proposals.reindex(columns=ordered_columns)
    return site_proposals

@orca.table('target_vacancies', cache=True)
def target_vacancies():
    vacancies = pd.read_csv('./data/target_vacancies.csv').\
        set_index('building_type_id')
    return vacancies


# Dictionary of variables to generate output indicators and charts
def creates_main_dicts():
    dict = {'total': {'households': 'Total households',
                      'jobs': 'Total jobs'},
            'sum': {
                'residential_units': 'Total residential units in buildings',
                'residential_sqft':
                    'Total residential area in buildings (sqft)',
                'non_residential_sqft':
                    'Total non residential sqft in buildings',
                'job_spaces': 'Total job spaces in buildings',
                'residential_units': 'Total number of residential units',
                'acres': 'Total area (acres)',
                'persons': 'Total persons in households',
                'workers': 'Total workers in households',
                'children': 'Total children in households',
                'cars': 'Total vehicles in households',
                'income': 'Total annual income from households',
                'recent_mover':
                    'Total households that moved within last 5 yrs'},
            'mean': {
                'non_residential_sqft':
                    'Average non residential sqft in buildings',
                'sqft_per_unit': 'Average area per residential unit in sqft',
                'sqft_per_unit_ave_800_linear':
                    'Average area per residential unit in sqft within 800m '
                    'along the auto street network (using flat decay)',
                'job_spaces': 'Average job spaces in buildings',
                'year_built': 'Average year of construction of buildings',
                'sector_id': 'Average job sector id',
                'acres': 'Average parcel area (acres)',
                'persons': 'Average persons in households',
                'workers': 'Average workers in households',
                'children': 'Average children in households',
                'cars': 'Average vehicles in households',
                'income': 'Average household annual income',
                'age_of_head': 'Average age of the household head',
                'x': 'Average x coordinate of parcels',
                'y': 'Average y coordinate of parcels',
                'value_per_unit': 'Average assessed value per unit',
                'value_per_sqft': 'Average assessed value per sqft of area'},
            'median': {
                   'building_type_id': 'Median building type id',
                   'income_quartile': 'Median income quartile',
                   'tenure': 'Median tenure code of households',
                   'race_of_head': 'Median race code of head of household',
                   'sector_id': 'Median job sector id'},
            'other': {'density_buildings': 'Density of buildings',
                  'density_households': 'Density of households',
                  'density_jobs': 'Density of jobs',
                  'ratio_jobs_to_households': 'Job-housing balance',
                  'ratio_workers_to_persons': 'Ratio of workers to persons',
                  'ratio_households_to_residential_units':
                    'Residential occupancy rate',
                  'residential_vacancy_rate':
                    'Total residential vacancy rate',
                  'non_residential_vacancy_rate':
                    'Total non residential vacancy rate',
                  'remaining_nonresidential_sqft_capacity':
                    'Total remaining non residential sqft capacity',
                  'remaining_residential_unit_capacity':
                    'Total remaining residential unit capacity',
                  'ave_annual_rent_sqft_400m':'Average annual rent per sqft '
                    'within 400m along the auto street network (flat decay)',
                  'ave_annual_office_rent_sqft_800m':'Average annual office '
                    'rent per sqft within 800m along the auto street network '
                    '(using flat decay)',
                  'ave_annual_industrial_rent_sqft_800m':'Average annual '
                    'industrial rent per sqft within 800m along the auto '
                    'street network (using flat decay)'}}
    custom_dict = {'jobs_sector_id':
                        {'data_name': 'Total jobs',
                        'aggregation_name': 'sector id'},
                    'households_income_quartile':
                        {'data_name': 'Total households',
                        'aggregation_name': 'income quartile'},
                    'households_age_of_head_quartile':
                        {'data_name': 'Total households',
                        'aggregation_name': 'age of head quartile'},
                    'households_recent_mover_income':
                        {'data_name': 'Total households that moved within last'
                                 ' 5 years',
                        'aggregation_name': 'income quartile (1 = lowest '
                                        'quartile, 2 = all others)'},
                    'buildings_repm_id':
                        {'data_name': 'Total buildings',
                        'aggregation_name': 'representative building type'}}
    prop_vars = {'households': ['persons', 'race_of_head', 'workers',
                                'children','cars', 'tenure', 'recent_mover',
                                'income_quartile'],
                 'jobs': ['sector_id'],
                 'buildings': ['building_type_id']}
    uses = ['retail', 'industrial','sf_detached', 'duplex_townhome',
            'multifamily', 'office']
    return dict, custom_dict, prop_vars, uses


def adds_dict_proportions(prop_vars, dict):
    prop = {}
    for agent in prop_vars:
        vars = prop_vars[agent]
        agents = orca.get_table(agent)
        for var in vars:
            agents_by_cat = agents[var].value_counts()
            cats_to_measure = agents_by_cat[agents_by_cat > 500].index.values
            for cat in cats_to_measure:
                new_var = var + '_' + str(cat)
                desc = 'Proportion of ' + agent + ' with ' + var + \
                       ' equal to ' + str(cat)
                prop[new_var] = desc
    dict['prop'] = prop
    return dict

def adds_derived_vars_dict(dict, uses):
    new_dict = {}
    derived_vars = {'total': ['households', 'jobs'],
                    'sum': dict['sum'].keys(),
                    'mean': dict['mean'].keys(),
                    'median': dict['median'].keys(),
                    'prop': dict['prop'].keys()}
    for agg in ['total', 'sum', 'mean', 'median', 'prop','other']:
        for var in dict[agg]:
            if agg != 'other':
                new_var = agg + '_' + var
            else:
                new_var = var
            new_dict[new_var] = dict[agg][var]
    for use in uses:
        var = 'mean_pred_' + use + '_price'
        new_var = var + '_ave_800_linear'
        new_dict[var] = 'Average predicted ' + use + ' price per sqft'
        method =' within 800m along the auto street network (using flat decay)'
        new_dict[new_var] = new_dict[var] + method

    for dist in [400, 1200, 2000, 2800, 3600]:
        for method in ['linear', 'flat']:
            for agg in ['total', 'sum', 'mean', 'prop']:
                for var in derived_vars[agg]:
                    new_var = agg + '_' + var + '_ave_' + str(
                        dist) + '_' + method
                    desc = 'Log of average within ' + str(dist/1000) + \
                           'km along the auto street network (' + method + \
                           ' decay) of: ' + \
                           dict[agg][var].strip('Log of ').capitalize()
                    new_dict[new_var] = desc

                    new_var = 'without_log_' + new_var
                    desc = 'Average within ' + str(dist / 1000) + \
                           'km along the auto street network (' + method + \
                           ' decay) of: ' + dict[agg][var]
                    new_dict[new_var] = desc
            for agg in ['total', 'sum']:
                for var in derived_vars[agg]:
                    new_var = agg + '_' + var + '_sum_' + str(
                        dist) + '_' + method
                    desc = 'Log of sum within ' + str(dist/1000) + \
                           'km along the auto street network (' + method + \
                           ' decay) of: ' + \
                           dict[agg][var].strip('Log of ').capitalize()
                    new_dict[new_var] = desc

                    new_var = 'without_log_' + new_var
                    desc = 'Sum within ' + str(dist / 1000) + \
                            'km along the auto street network (' + method + \
                            ' decay) of: ' + dict[agg][var]
                    new_dict[new_var] = desc
    return new_dict

@orca.injectable('dictionary')
def dictionary():
    new_dict = {}
    dict, custom_dict, prop_vars, uses = creates_main_dicts()
    dict = adds_dict_proportions(prop_vars, dict)
    new_dict = adds_derived_vars_dict(dict, uses)
    full_dict = {'var_dict': new_dict}
    full_dict['custom_var_dict'] = custom_dict
    return full_dict

