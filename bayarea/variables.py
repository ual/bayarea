import copy
import orca
import numpy as np
import pandas as pd

from urbansim.utils import misc
from variable_generators import generators

from lcog import datasources


@orca.column('households', cache=False)
def node_id(households, parcels):
    return misc.reindex(parcels.node_id, households.parcel_id)


@orca.column('jobs', cache=False)
def node_id(jobs, parcels):
    return misc.reindex(parcels.node_id, jobs.parcel_id)


@orca.column('buildings', cache=False)
def node_id(buildings, parcels):
    return misc.reindex(parcels.node_id, buildings.parcel_id)


@orca.column('buildings', cache=False)
def ave_annual_rent_sqft_400m(buildings, craigslist, net):
    net.set(craigslist.node_id, variable = craigslist.rent_sqft)
    results = net.aggregate(400, type='ave', decay='flat') * 12 #annualize
    return misc.reindex(results, buildings.node_id)


@orca.column('buildings')
def all_buildings(buildings):
    return pd.Series(np.ones(len(buildings)).astype('int32'), index=buildings.index)


def agg_var_building_type(geography, geography_id, var, buildingtype):

    """
    Register parcel or zone variable by building type with orca.
    Parameters
    ----------
    geography: str
        Name of the larger geography to summarize the building variable ('parcels' or 'zones')
    geography_id: str
        Unique identifier of the geography used to summarize ('parcel_id' or 'zone_id)
    var: str
        Variable that will be aggregated ('residential_units' or 'non_residential_sqft')
    buildingtype: int
        Numeric code for the building type stored in building_type_id

    Returns
    -------
    func : function
    """
    var_name = 'sum_' + var + '_' + str(buildingtype)
    @orca.column(geography, var_name, cache=True, cache_scope='step')
    def func():
        buildings = orca.get_table('buildings').to_frame(['building_type_id','parcel_id', 'residential_units', 'non_residential_sqft'])
        parcel_zones = orca.get_table('parcels').to_frame(['parcel_id', 'zone_id'])
        buildings = pd.merge(buildings, parcel_zones, on='parcel_id', how='left')
        buildings_iter = buildings[buildings['building_type_id'] == buildingtype].copy()
        values = buildings_iter[var].groupby(buildings_iter[geography_id]).sum().fillna(0)
        locations_index = orca.get_table(geography).index
        series = pd.Series(data=values, index=locations_index)
        series = series.fillna(0)
        return series
    return func

btype_columns = ['building_type_id', 'is_residential', 'is_non_residential']
btypes = orca.get_table('building_types').to_frame(btype_columns).reset_index()
res_types = btypes[btypes['is_residential'] == True].building_type_id
nonres_types = btypes[btypes['is_non_residential'] == True].building_type_id
geographic_types = ['parcels', 'zones']
vars = ['residential_units', 'non_residential_sqft']

for geography in geographic_types:
    if geography == 'parcels':
        geography_id = 'parcel_id'
    else:
        geography_id = 'zone_id'
    for var in vars:
        if var == 'residential_units':
            for buildingtype in res_types:
                agg_var_building_type(geography, geography_id, var, buildingtype)
        else:
            for buildingtype in nonres_types:
                agg_var_building_type(geography, geography_id, var, buildingtype)


@orca.column('zones', cache=True, cache_scope='step')
def residential_vacancy_rate(zones):
    zones = zones.to_frame(['zone_id', 'total_households', 'sum_residential_units'])
    zones['residential_vacancy_rate'] =  1 - zones['total_households'] / zones['sum_residential_units']
    zones.loc[zones['residential_vacancy_rate'] < 0, 'residential_vacancy_rate'] = 0
    zones = zones.fillna(0)
    return zones.residential_vacancy_rate

@orca.column('zones', cache=True, cache_scope='step')
def non_residential_vacancy_rate(zones):
    zones = zones.to_frame(['zone_id', 'total_jobs', 'sum_job_spaces'])
    zones['non_residential_vacancy_rate'] = 1 - zones['total_jobs'] / zones['sum_job_spaces']
    zones.loc[zones['non_residential_vacancy_rate'] < 0, 'non_residential_vacancy_rate'] = 0
    zones = zones.fillna(0)
    return zones.non_residential_vacancy_rate

@orca.column('parcels', cache=True, cache_scope='step')
def remaining_residential_unit_capacity(parcels):
    parcels=calc_remaining_residential_capacity()
    parcels.loc[parcels['remaining_residential_unit_capacity'] < 0, 'remaining_residential_unit_capacity'] = 0
    parcels = parcels.fillna(0)
    return parcels.remaining_residential_unit_capacity


@orca.column('zones', cache=True, cache_scope='step')
def remaining_residential_unit_capacity(zones):
    parcels = calc_remaining_residential_capacity()
    zones = parcels['remaining_residential_unit_capacity'].groupby(parcels['zone_id']).sum().reset_index()
    zones.loc[zones['remaining_residential_unit_capacity'] < 0, 'remaining_residential_unit_capacity'] = 0
    zones = zones.fillna(0)
    return zones.remaining_residential_unit_capacity


@orca.column('parcels', cache=True, cache_scope='step')
def remaining_nonresidential_sqft_capacity(parcels):
    parcels=calc_remaining_nonresidential_capacity()
    parcels.loc[parcels['remaining_nonresidential_sqft_capacity'] < 0, 'remaining_nonresidential_sqft_capacity'] = 0
    parcels = parcels.fillna(0)
    return parcels.remaining_nonresidential_sqft_capacity


@orca.column('zones', cache=True, cache_scope='step')
def remaining_nonresidential_sqft_capacity(zones):
    parcels = calc_remaining_nonresidential_capacity()
    zones = parcels['remaining_nonresidential_sqft_capacity'].groupby(parcels['zone_id']).sum().reset_index()
    zones.loc[zones['remaining_nonresidential_sqft_capacity']<0,'remaining_nonresidential_sqft_capacity']=0
    zones = zones.fillna(0)
    return zones.remaining_nonresidential_sqft_capacity


def calc_remaining_residential_capacity():
    parcels = orca.get_table('parcels').to_frame(['parcel_id', 'zone_id', 'zoning_id', 'acres', 'sum_residential_units'])
    zone_types = orca.get_table('zone_types').to_frame(['zoning_id', 'max_dua'])
    allowable_buildings = orca.get_table('allowable_building_types').to_frame(['zoning_id','building_type_id'])
    df = pd.merge(zone_types, allowable_buildings, on='zoning_id', how='left')
    btype_columns = ['building_type_id','is_residential']
    btypes = orca.get_table('building_types').to_frame(btype_columns).reset_index()
    res_types = btypes[btypes['is_residential'] == True].building_type_id
    df.loc[df.building_type_id.isin(res_types), 'residential'] = 1
    df = df['residential'].groupby(df['zoning_id']).sum().reset_index()
    zone_types = pd.merge(zone_types, df, on='zoning_id', how='left').fillna(0)
    parcels = pd.merge(parcels, zone_types, on='zoning_id', how='left')
    parcels.loc[parcels.residential>0,'remaining_residential_unit_capacity'] = parcels['acres']*parcels['max_dua'] \
                                                                                 - parcels['sum_residential_units']
    parcels=parcels.fillna(0)
    return parcels

def calc_remaining_nonresidential_capacity():
    parcels = orca.get_table('parcels').to_frame(['parcel_id', 'zone_id', 'zoning_id', 'acres', 'sum_non_residential_sqft'])
    zone_types = orca.get_table('zone_types').to_frame(['zoning_id', 'max_far'])
    allowable_buildings = orca.get_table('allowable_building_types').to_frame(['zoning_id', 'building_type_id'])
    df = pd.merge(zone_types, allowable_buildings, on='zoning_id', how='left')
    btype_columns = ['building_type_id', 'is_non_residential']
    btypes = orca.get_table('building_types').to_frame(btype_columns).reset_index()
    nonres_types = btypes[btypes['is_non_residential'] == True].building_type_id
    df.loc[df.building_type_id.isin(nonres_types), 'non_residential'] = 1
    df = df['non_residential'].groupby(df['zoning_id']).sum().reset_index()
    zone_types = pd.merge(zone_types, df, on='zoning_id', how='left').fillna(0)
    parcels = pd.merge(parcels, zone_types, on='zoning_id', how='left')
    parcels.loc[parcels.non_residential > 0, 'remaining_nonresidential_sqft_capacity'] = parcels['acres'] \
                                                                                         *43560 * parcels['max_far']                                                                                 - parcels['sum_non_residential_sqft']
    parcels = parcels.fillna(0)

    return parcels

@orca.column('buildings', cache=True)
def repm_id(buildings):
    buildings = orca.get_table('buildings').to_frame(['building_type_id'])
    buildings['repm_id'] = 'na'

    # Retail
    retail_btypes = [4200, 4210, 4220, 4230, 4240, 4250, 4260, 4290, 4300, 4310]
    buildings.repm_id[buildings.building_type_id.isin(retail_btypes)] = 'retail'

    # Industrial
    industrial_btypes = [5100, 5200]
    buildings.repm_id[buildings.building_type_id.isin(industrial_btypes)] = 'industrial'

    # Office
    buildings.repm_id[buildings.building_type_id == 4100] = 'office'

    # Residential
    buildings.repm_id[buildings.building_type_id == 1110] = 'res_sf_detached'
    duplex_townhome_btypes = [1121, 1122]
    buildings.repm_id[buildings.building_type_id.isin(duplex_townhome_btypes)] = 'duplex_townhome'
    mf_btypes = [1210, 1220]
    buildings.repm_id[buildings.building_type_id.isin(mf_btypes)] = 'multifamily'

    # Educational
    educ_btypes = [6110, 6120, 6130, 6140, 6150, 6160]
    buildings.repm_id[buildings.building_type_id.isin(educ_btypes)] = 'educational'

    # Other
    other_btypes = [4320, 4400, 4900, 6300, 6400, 8000, 9000, -1]
    buildings.repm_id[buildings.building_type_id.isin(other_btypes)] = 'other'

    return buildings.repm_id


@orca.column('buildings', cache=True)
def job_spaces(buildings):
    df_per_job = orca.get_table('building_sqft_per_job').to_frame().set_index('building_type_id')
    sqft_per_job = orca.get_table('buildings').building_type_id.map(df_per_job.area_per_job)
    spaces = (buildings.non_residential_sqft / sqft_per_job).astype('int')
    spaces.fillna(0)
    return spaces


@orca.column('households', cache=True)
def income_quartile(households):
    s = pd.Series(pd.qcut(households.income, 4, labels=False),
                  index=households.index)
    # e.g. convert income quartile from 0-3 to 1-4
    s = s.add(1)
    return s


@orca.column('households', cache=True)
def age_of_head_quartile(households):
    s = pd.Series(pd.qcut(households.age_of_head, 4, labels=False),
                  index=households.index)
    s = s.add(1)
    return s


@orca.column('households', cache=True)
def recent_mover_income(households):
    households = orca.get_table('households').to_frame(['recent_mover', 'income_quartile'])
    households['recent_mover_income'] = 0
    households[(households.recent_mover == 1) & (households.income_quartile == 1)] = 1
    households[(households.recent_mover == 1) & (households.income_quartile > 1)] = 2
    return households.recent_mover_income


@orca.column('buildings', cache=True)
def sqft_per_unit(buildings):
    residential_sqft = buildings.residential_sqft
    residential_units = buildings.residential_units
    sqft_per_unit = residential_sqft / residential_units
    return sqft_per_unit.fillna(0).replace(np.inf, 0)


@orca.column('buildings', cache=True)
def value_per_unit(buildings):
    improvement_value = buildings.improvement_value
    residential_units = buildings.residential_units
    value_per_unit = improvement_value / residential_units
    return value_per_unit.fillna(0).replace(np.inf, 0)


@orca.column('buildings', cache=True)
def value_per_sqft(buildings):
    improvement_value = buildings.improvement_value
    non_residential_sqft = buildings.non_residential_sqft
    value_per_sqft = improvement_value / non_residential_sqft
    return value_per_sqft.fillna(0).replace(np.inf, 0)


@orca.column('households', cache=False)
def parcel_id(households, buildings):
    return misc.reindex(buildings.parcel_id, households.building_id)


@orca.column('jobs', cache=False)
def parcel_id(jobs, buildings):
    return misc.reindex(buildings.parcel_id, jobs.building_id)


geographic_levels = [('parcels', 'parcel_id')]
# Define parcel -> agent/building disaggregation vars
for base_geography in ['households', 'jobs', 'buildings']:
    for geography in geographic_levels:
        geography_name = geography[0]
        geography_id = geography[1]
        if geography_name != base_geography:
            for var in orca.get_table(geography_name).columns:
                generators.make_disagg_var(geography_name, base_geography, var,
                                           geography_id, name_based_on_geography=False)

# Generate variables to serve as a pool of variables for location
# choice model to select from

aggregation_functions = ['mean', 'median', 'std', 'sum']

geographic_levels = copy.copy(orca.get_injectable('aggregate_geos'))
geographic_levels['parcels'] = 'parcel_id'

variables_to_aggregate = {
    'households': ['persons', 'income', 'race_of_head', 'age_of_head',
                   'workers', 'children', 'cars', 'hispanic_head', 'tenure',
                   'recent_mover', 'income_quartile'],
    'jobs':       ['sector_id'],
    'parcels':    ['acres', 'x', 'y', 'land_value', 'proportion_undevelopable'],
    'buildings':  ['building_type_id', 'residential_units', 'non_residential_sqft', 'year_built', 
                   'value_per_unit', 'sqft_per_unit', 'job_spaces']
                         }

discrete_variables = {
    'households': ['persons', 'race_of_head', 'workers', 'children',
                   'cars', 'hispanic_head', 'tenure', 'recent_mover', 'income_quartile'],
    'jobs': ['sector_id'],
    'buildings': ['building_type_id']
    }
sum_vars = ['persons', 'workers', 'children', 'cars', 'hispanic_head',
            'recent_mover', 'acres', 'land_value', 'residential_units',
            'non_residential_sqft', 'job_spaces']

geog_vars_to_dummify = orca.get_injectable('aggregate_geos').values()

generated_variables = set([])

orca.add_column('parcels', 'sum_acres', orca.get_table('parcels').acres) # temporary

for agent in variables_to_aggregate.keys():
    for geography_name, geography_id in geographic_levels.items():
        if geography_name != agent:

            # Define size variables
            generators.make_size_var(agent, geography_name, geography_id)
            generated_variables.add('total_' + agent)

            # Define attribute variables
            variables = variables_to_aggregate[agent]
            for var in variables:
                for aggregation_function in aggregation_functions:
                    if aggregation_function == 'sum':
                        if var in sum_vars:
                            generators.make_agg_var(agent, geography_name,
                                                    geography_id,
                                                    var, aggregation_function)
                            generated_variables.add(
                                aggregation_function + '_' + var)

                    else:
                        generators.make_agg_var(agent, geography_name,
                                                geography_id, var,
                                                aggregation_function)
                        generated_variables.add(
                            aggregation_function + '_' + var)

# Define prop_X_X variables
for agent in discrete_variables.keys():
    agents = orca.get_table(agent)
    discrete_vars = discrete_variables[agent]
    for var in discrete_vars:
        agents_by_cat = agents[var].value_counts()
        cats_to_measure = agents_by_cat[agents_by_cat > 500].index.values
        for cat in cats_to_measure:
            for geography_name, geography_id in geographic_levels.items():
                generators.make_proportion_var(agent, geography_name,
                                               geography_id, var, cat)
                generated_variables.add('prop_%s_%s' % (var, int(cat)))

# Making proportion by geography with global building types
agent = 'buildings'
var = 'repm_id'
cats_to_measure = ['res_sf_detached', 'duplex_townhome', 'multifamily', 
                    'retail', 'industrial', 'office', 'educational']
for cat in cats_to_measure:
    var_name = 'prop_repm_id_{}'.format(cat)
    for geography_name, geography_id in geographic_levels.items():
        generators.make_proportion_var(agent, geography_name, geography_id, var, cat)
        generated_variables.add('prop_%s_%s' % (var, cat))

# Define ratio variables
for geography_name in geographic_levels.keys():

    # Jobs-housing balance
    generators.make_ratio_var('jobs', 'households', geography_name)
    generated_variables.add('ratio_jobs_to_households')

    # # workers-persons ratio
    generators.make_ratio_var('workers', 'persons', geography_name, prefix1 = 'sum', prefix2 = 'sum')
    generated_variables.add('ratio_workers_to_persons')

    # Residential occupancy rate
    generators.make_ratio_var('households', 'residential_units', geography_name, prefix2 = 'sum')
    generated_variables.add('ratio_households_to_residential_units')

    # Density
    for agent in discrete_variables.keys():
        generators.make_density_var(agent, geography_name)
        generated_variables.add('density_%s' % agent)


for geog_var in geog_vars_to_dummify:
    geog_ids = np.unique(orca.get_table('parcels')[geog_var])
    if len(geog_ids) < 50:
        for geog_id in geog_ids:
            generators.make_dummy_variable('parcels', geog_var, geog_id)


#### Accessibility variable creation functions ####

def register_pandana_access_variable(column_name, onto_table, variable_to_summarize,
                                     distance, agg_type='sum', decay='linear', log=True):
    """
    Register pandana accessibility variable with orca.
    Parameters
    ----------
    column_name : str
        Name of the orca column to register this variable as.
    onto_table : str
        Name of the orca table to register this table with.
    variable_to_summarize : str
        Name of the onto_table variable to summarize.
    distance : int
        Distance along the network to query.
    agg_type : str
        Pandana aggregation type.
    decay : str
        Pandana decay type.
    Returns
    -------
    column_func : function
    """
    @orca.column(onto_table, column_name, cache=True, cache_scope='iteration')
    def column_func():
        net = orca.get_injectable('net')  # Get the pandana network
        table = orca.get_table(onto_table).to_frame(['node_id', variable_to_summarize])
        net.set(table.node_id,  variable=table[variable_to_summarize])
        try:
            results = net.aggregate(distance, type=agg_type, decay=decay)
        except:
            results = net.aggregate(distance, type=agg_type, decay=decay)  # import pdb; pdb.set_trace()
        if log:
            results = results.apply(eval('np.log1p'))
        return misc.reindex(results, table.node_id)
    return column_func


def register_skim_access_variable(column_name, variable_to_summarize, impedance_measure,
                                  distance, log=False):
    """
    Register skim-based accessibility variable with orca.
    Parameters
    ----------
    column_name : str
        Name of the orca column to register this variable as.
    impedance_measure : str
        Name of the skims column to use to measure inter-zone impedance.
    variable_to_summarize : str
        Name of the zonal variable to summarize.
    distance : int
        Distance to query in the skims (e.g. 30 minutes travel time).
    Returns
    -------
    column_func : function
    """
    @orca.column('zones', column_name, cache=True, cache_scope='iteration')
    def column_func(zones, travel_data):
        results = misc.compute_range(travel_data.to_frame(), zones.get_column(variable_to_summarize),
                                  impedance_measure, distance, agg=np.sum)
        if log:
            results = results.apply(eval('np.log1p'))

        if len(results) < len(zones):
            results = results.reindex(zones.index).fillna(0)

        return results
    return column_func


# Calculate pandana-based accessibility variable
distances = range(400, 5000, 400)
print(distances)
agg_types = ['ave', 'sum']
decay_types = ['linear', 'flat']
variables_to_aggregate = ['sum_children',
                            'sum_persons',
                            'sum_workers',
                            'sum_residential_units',
                            'sum_non_residential_sqft',
                            'total_households',
                            'total_jobs',
                            'sum_residential_units',
                            'sum_non_residential_sqft']  # add building vars here

variables_to_aggregate_avg_only = ['prop_race_of_head_1',
                                    'prop_race_of_head_9',
                                    'mean_age_of_head',
                                    'mean_children',
                                    'mean_income',
                                    'mean_workers',
                                    'mean_value_per_unit',
                                    'mean_non_residential_sqft']  # Add building/job vars here
access_vars = []
for distance in distances:
    for decay in decay_types:
        for variable in variables_to_aggregate:
            for agg_type in agg_types:
                var_name = '_'.join([variable, agg_type, str(distance), decay])
                access_vars.append(var_name)
                register_pandana_access_variable(var_name, 'parcels', variable, distance, agg_type=agg_type, decay=decay)
                not_log = 'without_log_' + var_name
                register_pandana_access_variable(not_log, 'parcels', variable, distance, agg_type=agg_type, decay=decay, log=False)
                generated_variables.add(var_name)

        for variable in variables_to_aggregate_avg_only:
            var_name = '_'.join([variable, 'ave', str(distance), decay])
            access_vars.append(var_name)
            register_pandana_access_variable(var_name, 'parcels', variable, distance, agg_type='ave', decay=decay)
            not_log = 'without_log_' + var_name
            register_pandana_access_variable(not_log, 'parcels', variable, distance, agg_type='ave', decay=decay, log=False)
            generated_variables.add(var_name)

# Network-based price aggregations for proforma input
price_cols = ['pred_sf_detached_price', 'pred_duplex_townhome_price',
            'pred_multifamily_price', 'pred_office_price', 'pred_retail_price',
            'pred_industrial_price']
for price_col in price_cols:
    generators.make_agg_var('buildings', 'parcels',
                            'parcel_id', price_col,
                            'mean')
    register_pandana_access_variable('%s_ave_800_linear' % price_col, 'parcels', 'mean_%s' % price_col,
                                     800, agg_type='ave', decay='flat', log=False)

register_pandana_access_variable('mean_sqft_per_unit_ave_800_linear', 'parcels', 'mean_sqft_per_unit',
                                 800, agg_type='ave', decay='flat', log=False)

# Calculate skim-based accessibility variable
variables_to_aggregate = ['total_jobs', 'sum_persons']
skim_access_vars = []
# Transit skim variables
travel_times = [5, 10, 15, 25]
for time in travel_times:
    for variable in variables_to_aggregate:
        var_name = '_'.join([variable, str(time), 'am_peak_travel_time'])
        skim_access_vars.append(var_name)
        register_skim_access_variable(var_name, variable, 'am_peak_travel_time', time)
        generated_variables.add(var_name)

        var_name = '_'.join([variable, str(time), 'md_offpeak_travel_time'])
        skim_access_vars.append(var_name)
        register_skim_access_variable(var_name, variable, 'md_offpeak_travel_time', time)
        generated_variables.add(var_name)


# Disaggregate higher-level variables to the building level
for base_geography in ['buildings']:
    for geography_name, geography_id in geographic_levels.items():
        if geography_name != base_geography:
            for var in orca.get_table(geography_name).columns:
                generators.make_disagg_var(geography_name, base_geography, var,
                                           geography_id, name_based_on_geography=True)


# Create logged version of all building variables for estimation
def register_ln_variable(table_name, column_to_ln):
    """
    Register logged variable with orca.
    Parameters
    ----------
    table_name : str
        Name of the orca table that this column is part of.
    column_to_ln : str
        Name of the orca column to log.
    Returns
    -------
    column_func : function
    """
    new_col_name = 'ln_' + column_to_ln

    @orca.column(table_name, new_col_name, cache=True, cache_scope='iteration')
    def column_func():
        return np.log1p(orca.get_table(table_name)[column_to_ln])
    return column_func


for var in orca.get_table('buildings').columns:
    register_ln_variable('buildings', var)


# Building type dummies
@orca.column('buildings', cache=True)
def is_office():
    series = (orca.get_table('buildings').building_type_id.isin([4100, 2121, 2122])).astype(int)
    return series


@orca.column('buildings',  cache=True)
def is_warehouse():
    return (orca.get_table('buildings').building_type_id.isin([5100])).astype(int)


@orca.column('buildings',  cache=True)
def is_industrial():
    return (orca.get_table('buildings').building_type_id.isin([5100, 5200])).astype(int)


@orca.column('buildings',  cache=True)
def is_multifamily():
    return (orca.get_table('buildings').building_type_id.isin([1210, 1220])).astype(int)


# Building Age dummies
@orca.column('buildings', cache=True)
def built_before_1950():
    return (orca.get_table('buildings').year_built < 1950).astype(int)


@orca.column('buildings', cache=True)
def built_after_2000():
    return (orca.get_table('buildings').year_built > 2000).astype(int)


@orca.column('buildings', cache=True)
def land_value_per_acre():
    return (orca.get_table('buildings').land_value / 
            orca.get_table('buildings').acres).fillna(0)


register_ln_variable('buildings', 'land_value_per_acre')

# HOUSEHOLDS VARIABLES
@orca.column('households', cache=True)
def income_quartile_1():
    return (orca.get_table('households').income_quartile == 1).astype(int)


@orca.column('households', cache=True)
def income_quartile_2():
    return (orca.get_table('households').income_quartile == 2).astype(int)


@orca.column('households', cache=True)
def income_quartile_3():
    return (orca.get_table('households').income_quartile == 3).astype(int)

@orca.column('households', cache=True)
def no_children():
    return (orca.get_table('households').children == 0).astype(int)


@orca.column('households', cache=True)
def has_children():
    return (orca.get_table('households').children > 0).astype(int)


@orca.column('households', cache=True)
def no_workers():
    return (orca.get_table('households').workers == 0).astype(int)


@orca.column('households', cache=True)
def race_notwhite():
    return (orca.get_table('households').race_of_head > 1).astype(int)


@orca.column('households', cache=True)
def race_white():
    return (orca.get_table('households').race_of_head == 1).astype(int)


@orca.column('households', cache=True)
def race_black():
    return (orca.get_table('households').race_of_head == 2).astype(int)


@orca.column('households', cache=True)
def race_asian():
    return (orca.get_table('households').race_of_head == 6).astype(int)

@orca.column('households', cache=True)
def no_hispanic_head():
    return (orca.get_table('households').race_of_head != 1).astype(int)


@orca.column('households', cache=True)
def zero_carowner():
    return (orca.get_table('households').cars == 0).astype(int)


@orca.column('households', cache=True)
def carowner():
    return (orca.get_table('households').cars > 0).astype(int)


@orca.column('households', cache=True)
def income_less25K():
    return (orca.get_table('households')['income'] < 25000).astype(int)


@orca.column('households', cache=True)
def income_25to45K():
    return ((orca.get_table('households')['income'] >= 25000) & (orca.get_table('households')['income'] < 45000)).astype(int)


@orca.column('households', cache=True)
def income_45to70K():
    return ((orca.get_table('households')['income'] >= 45000) & (orca.get_table('households')['income'] < 70000)).astype(int)


@orca.column('households', cache=True)
def income_70to90K():
    return ((orca.get_table('households')['income'] >= 70000) & (orca.get_table('households')['income'] < 90000)).astype(int)


@orca.column('households', cache=True)
def income_90to110K():
    return ((orca.get_table('households')['income'] >= 90000) & (orca.get_table('households')['income'] < 110000)).astype(int)


@orca.column('households', cache=True)
def income_110to150K():
    return ((orca.get_table('households')['income'] >= 110000) & (orca.get_table('households')['income'] < 150000)).astype(int)


@orca.column('households', cache=True)
def income_more150K():
    return (orca.get_table('households')['income'] >= 150000).astype(int)


@orca.column('households', cache=True)
def ratio_income_persons():
    return orca.get_table('households')['income'] / orca.get_table('households')['persons']


@orca.column('households', cache=True)
def tenure_rent():
    return (orca.get_table('households').tenure == 2).astype(int)


@orca.column('households', cache=True)
def tenure_own():
    return (orca.get_table('households').tenure == 1).astype(int)


@orca.column('households', cache=True)
def living_alone():
    return (orca.get_table('households').persons == 1).astype(int)


@orca.column('households', cache=True)
def hh_size_2():
    return (orca.get_table('households').persons == 2).astype(int)


@orca.column('households', cache=True)
def hh_size_3():
    return (orca.get_table('households').persons == 3).astype(int)


@orca.column('households', cache=True)
def hh_size_4():
    return (orca.get_table('households').persons == 4).astype(int)


@orca.column('households', cache=True)
def hh_size_more4():
    return (orca.get_table('households').persons > 4).astype(int)


@orca.column('households', cache=True)
def age_head_less40():
    return (orca.get_table('households').age_of_head <= 40).astype(int)


@orca.column('households', cache=True)
def age_head_more40_age():
    return (orca.get_table('households').age_of_head > 40).astype(int) * orca.get_table('households').age_of_head


@orca.column('households', cache=True)
def ratio_cars_workers():
    ratio = orca.get_table('households')['cars'] / orca.get_table('households')['workers']
    ratio.replace({np.inf : 0, -np.inf : 0},inplace=True)
    ratio.fillna(0, inplace=True)
    return ratio

ln_vars = ['age_of_head', 'persons', 'workers',
            'income', 'cars',
            'ratio_income_persons', 'ratio_cars_workers']

for lnv in ln_vars:
    register_ln_variable('households', lnv)


@orca.column('parcels', cache=False)
def total_yearly_rent(parcels):
    parcels = parcels.to_frame(['sum_residential_units', 'sum_non_residential_sqft',
                                'mean_pred_sf_detached_price', 'mean_pred_duplex_townhome_price', 'mean_pred_multifamily_price',
                                'mean_pred_office_price', 'mean_pred_retail_price', 'mean_pred_industrial_price'])

    parcels[parcels < 0] = 0
    parcels['mean_resunit_price'] = parcels[['mean_pred_sf_detached_price', 'mean_pred_duplex_townhome_price', 'mean_pred_multifamily_price']].mean(axis=1)
    parcels['mean_nrsf_price'] = parcels[['mean_pred_office_price', 'mean_pred_retail_price', 'mean_pred_industrial_price']].mean(axis=1)

    res_price = parcels.mean_resunit_price * parcels.sum_residential_units
    nonres_price = parcels.mean_nrsf_price * parcels.sum_non_residential_sqft
    return (res_price + nonres_price) * .05 ## expressed in current annual rent


@orca.column('parcels', cache=True)
def developable_sqft(parcels):
    return (1 - (parcels.proportion_undevelopable / 100.0)) * parcels.acres * 43560


## Disagg from parcel to site proposals
parcel_cols = orca.get_table('parcels').columns
site_proposal_cols = orca.get_table('site_proposals').columns
for var in parcel_cols:
    if var not in site_proposal_cols:
        generators.make_disagg_var('parcels', 'site_proposals', var,
                                   'parcel_id', name_based_on_geography=False)


# Household_pums variables
@orca.column('households_pums', cache=True)
def income_quartile(households_pums):
    s = pd.Series(pd.qcut(households_pums.income, 4, labels=False),
                  index=households_pums.index)
    # e.g. convert income quartile from 0-3 to 1-4
    s = s.add(1)
    return s


@orca.column('households_pums', cache=True)
def income_quartile_1():
    return (orca.get_table('households_pums').income_quartile == 1).astype(int)


@orca.column('households_pums', cache=True)
def income_quartile_2():
    return (orca.get_table('households_pums').income_quartile == 2).astype(int)


@orca.column('households_pums', cache=True)
def income_quartile_3():
    return (orca.get_table('households_pums').income_quartile == 3).astype(int)


@orca.column('households_pums', cache=True)
def no_workers():
    return (orca.get_table('households_pums').workers == 0).astype(int)


@orca.column('households_pums', cache=True)
def race_notwhite():
    return (orca.get_table('households_pums').race_of_head > 1).astype(int)


@orca.column('households_pums', cache=True)
def race_white():
    return (orca.get_table('households_pums').race_of_head == 1).astype(int)


@orca.column('households_pums', cache=True)
def race_black():
    return (orca.get_table('households_pums').race_of_head == 2).astype(int)


@orca.column('households_pums', cache=True)
def race_asian():
    return (orca.get_table('households_pums').race_of_head == 6).astype(int)

@orca.column('households_pums', cache=True)
def no_hispanic_head():
    return (orca.get_table('households_pums').race_of_head != 1).astype(int)


@orca.column('households_pums', cache=True)
def zero_carowner():
    return (orca.get_table('households_pums').cars == 0).astype(int)


@orca.column('households_pums', cache=True)
def carowner():
    return (orca.get_table('households_pums').cars > 0).astype(int)


@orca.column('households_pums', cache=True)
def income_more150K():
    return (orca.get_table('households_pums')['income'] >= 150000).astype(int)


@orca.column('households_pums', cache=True)
def ratio_income_persons():
    return orca.get_table('households_pums')['income'] / orca.get_table('households_pums')['persons']


@orca.column('households_pums', cache=True)
def ratio_cars_workers():
    ratio = orca.get_table('households_pums')['cars'] / orca.get_table('households_pums')['workers']
    ratio.replace({np.inf : 0, -np.inf : 0},inplace=True)
    ratio.fillna(0, inplace=True)
    return ratio


@orca.column('households_pums', cache=True)
def tenure_rent():
    return (orca.get_table('households_pums').tenure == 2).astype(int)


@orca.column('households_pums', cache=True)
def tenure_own():
    return (orca.get_table('households_pums').tenure == 1).astype(int)


@orca.column('households_pums', cache=True)
def living_alone():
    return (orca.get_table('households_pums').persons == 1).astype(int)


ln_vars = ['age_of_head', 'persons', 'workers',
            'income', 'cars',
            'ratio_income_persons', 'ratio_cars_workers']

for lnv in ln_vars:
    register_ln_variable('households_pums', lnv)
