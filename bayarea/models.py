import orca
import pandas as pd
import pandana as pdna
from urbansim.models import transition
from urbansim_parcels import utils as parcel_utils
import json
from altair import Chart, X, Y, Axis
import altair as alt
import numpy as np
import yaml
from developer import proposal_select
from collections import OrderedDict
from urbansim.utils import networks

# ~ @orca.step()
# ~ def build_networks(parcels, nodes, edges, craigslist):
    # ~ nodes, edges = nodes.to_frame(), edges.to_frame()
    # ~ print('Number of nodes is %s.' % len(nodes))
    # ~ print('Number of edges is %s.' % len(edges))
    # ~ net = pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"],
                           # ~ edges[["weight"]])

    # ~ precompute_distance = 5000
    # ~ print('Precomputing network for distance %s.' % precompute_distance)
    # ~ print('Network precompute starting.')
    # ~ net.precompute(precompute_distance)
    # ~ print('Network precompute done.')

    # ~ parcels = parcels.local
    # ~ parcels['node_id'] = net.get_node_ids(parcels['x'], parcels['y'])
    # ~ orca.add_table("parcels", parcels)
    # ~ orca.add_injectable("net", net)

    # ~ craigslist = craigslist.local
    # ~ craigslist['node_id'] = net.get_node_ids(craigslist['longitude'], craigslist['latitude'])
    # ~ orca.add_table('craigslist', craigslist)


@orca.step()
def initialize_network_small():
    """
    This will be turned into a data loading template.
    """

    @orca.injectable('netsmall', cache=True)
    def build_networksmall(parcels, craigslist, nodessmall, edgessmall):
        netsmall = pdna.Network(nodessmall.x, nodessmall.y, edgessmall.u,
                                 edgessmall.v, edgessmall[['length']], twoway=True)
        netsmall.precompute(25000)
        
        parcels = parcels.local
        parcels['node_id_small'] = netsmall.get_node_ids(parcels['x'], parcels['y'])
        orca.add_table("parcels", parcels)
        orca.add_injectable("netsmall", netsmall)

        craigslist = craigslist.local
        craigslist['node_id_small'] = netsmall.get_node_ids(craigslist['longitude'], craigslist['latitude'])
        orca.add_table('craigslist', craigslist)


@orca.step()
def initialize_network_walk():
    """
    This will be turned into a data loading template.
    """

    @orca.injectable('netwalk', cache=False)
    def build_networkwalk(parcels, craigslist, nodeswalk, edgeswalk):
        netwalk = pdna.Network(nodessmall.x, nodessmall.y, edgessmall.u,
                                 edgessmall.v, edgessmall[['length']], twoway=True)
        netwalk.precompute(5000)
        
        parcels = parcels.local
        parcels['node_id_walk'] = netwalk.get_node_ids(parcels['x'], parcels['y'])
        orca.add_table("parcels", parcels)
        orca.add_injectable("netwalk", netwalk)

        craigslist = craigslist.local
        craigslist['node_id_walk'] = netwalk.get_node_ids(craigslist['longitude'], craigslist['latitude'])
        orca.add_table('craigslist', craigslist)



@orca.step()
def initialize_network_beam(parcels, craigslist):
    """
    This will be turned into a data loading template.
    """

    @orca.injectable('netbeam', cache=True)
    def build_networkbeam(nodesbeam, edgesbeam, parcels, craigslist):
        nodesbeam, edgesbeam = nodesbeam.to_frame(), edgesbeam.to_frame()
        print('Number of nodes is %s.' % len(nodesbeam))
        print('Number of edges is %s.' % len(edgesbeam))
        #nodesbeam = orca.get_table('nodesbeam',nodesbeam).local
        #edgesbeam = orca.get_table('edgeswalk',edgeswalk).local
        #edgesbeam = edgesbeam[
        #    (edgesbeam['hour'] == 8) & (edgesbeam['stat'] == 'AVG')]
        netbeam = pdna.Network(
            nodesbeam['lon'], nodesbeam['lat'], edgesbeam['from'],
            edgesbeam['to'], edgesbeam[['traveltime']], twoway=False)
        netbeam.precompute(1000)
        #return netbeam
        
        parcels = parcels.local
        parcels['node_id_beam'] = netbeam.get_node_ids(parcels['x'], parcels['y'])
        orca.add_table("parcels", parcels)
        orca.add_injectable("netbeam", netbeam)

        craigslist = craigslist.local
        craigslist['node_id_beam'] = netbeam.get_node_ids(craigslist['longitude'], craigslist['latitude'])
        orca.add_table('craigslist', craigslist)

@orca.step()
def network_aggregations_small(netsmall):
    """
    This will be turned into a network aggregation template.
    """
    nodessmall = networks.from_yaml(
        netsmall, 'network_aggregations_small.yaml')
    nodessmall = nodessmall.fillna(0)
    
    # new variables
    print('compute additional aggregation variables')
    nodessmall['pop_jobs_ratio_10000'] = (nodessmall['pop_10000'] / (nodessmall['jobs_10000'])).fillna(0)
    nodessmall['pop_jobs_ratio_25000'] = (nodessmall['pop_25000'] / (nodessmall['jobs_25000'])).fillna(0)
    # fill inf and nan with median
    nodessmall['pop_jobs_ratio_10000'] = nodessmall['pop_jobs_ratio_10000'].replace([np.inf, -np.inf], np.nan).fillna(
        nodessmall['pop_jobs_ratio_10000'].median)
    nodessmall['pop_jobs_ratio_25000'] = nodessmall['pop_jobs_ratio_25000'].replace([np.inf, -np.inf], np.nan).fillna(
        nodessmall['pop_jobs_ratio_25000'].median)
    
    # end of addition
    
    print(nodessmall.describe())
    orca.add_table('nodessmall', nodessmall)


@orca.step()
def network_aggregations_walk(netwalk):
    """
    This will be turned into a network aggregation template.
    """

    nodeswalk = networks.from_yaml(netwalk, 'network_aggregations_walk.yaml')
    nodeswalk = nodeswalk.fillna(0)
    
    # new variables
    print('compute additional aggregation variables')
    nodeswalk['prop_children_500_walk'] = ((nodeswalk['children_500_walk'] > 0).astype(int) / nodeswalk['hh_500_walk']).fillna(0)
    nodeswalk['prop_singles_500_walk'] = (nodeswalk['singles_500_walk'] / nodeswalk['hh_500_walk']).fillna(0)
    nodeswalk['prop_elderly_500_walk'] = (nodeswalk['elderly_hh_500_walk'] / nodeswalk['hh_500_walk']).fillna(0)
    nodeswalk['prop_black_500_walk'] = (nodeswalk['pop_black_500_walk'] / nodeswalk['pop_500_walk']).fillna(0)
    nodeswalk['prop_white_500_walk'] = (nodeswalk['pop_white_500_walk'] / nodeswalk['pop_500_walk']).fillna(0)
    nodeswalk['prop_asian_500_walk'] = (nodeswalk['pop_asian_500_walk'] / nodeswalk['pop_500_walk']).fillna(0)
    nodeswalk['prop_hisp_500_walk'] = (nodeswalk['pop_hisp_500_walk'] / nodeswalk['pop_500_walk']).fillna(0)
    nodeswalk['prop_rich_500_walk'] = (nodeswalk['rich_500_walk'] / nodeswalk['pop_500_walk']).fillna(0)
    nodeswalk['prop_poor_500_walk'] = (nodeswalk['poor_500_walk'] / nodeswalk['pop_500_walk']).fillna(0)

    nodeswalk['prop_children_1500_walk'] = ((nodeswalk['children_1500_walk'] > 0).astype(int)/nodeswalk['hh_1500_walk']).fillna(0)
    nodeswalk['prop_singles_1500_walk'] = (nodeswalk['singles_1500_walk'] / nodeswalk['hh_1500_walk']).fillna(0)
    nodeswalk['prop_elderly_1500_walk'] = (nodeswalk['elderly_hh_1500_walk'] / nodeswalk['hh_1500_walk']).fillna(0)
    nodeswalk['prop_black_1500_walk'] = (nodeswalk['pop_black_1500_walk'] / nodeswalk['pop_1500_walk']).fillna(0)
    nodeswalk['prop_white_1500_walk'] = (nodeswalk['pop_white_1500_walk'] / nodeswalk['pop_1500_walk']).fillna(0)
    nodeswalk['prop_asian_1500_walk'] = (nodeswalk['pop_asian_1500_walk'] / nodeswalk['pop_1500_walk']).fillna(0)
    nodeswalk['prop_hisp_1500_walk'] = (nodeswalk['pop_hisp_1500_walk'] / nodeswalk['pop_1500_walk']).fillna(0)
    nodeswalk['prop_rich_1500_walk'] = (nodeswalk['rich_1500_walk'] / nodeswalk['pop_1500_walk']).fillna(0)
    nodeswalk['prop_poor_1500_walk'] = (nodeswalk['poor_1500_walk'] / nodeswalk['pop_1500_walk']).fillna(0)

    nodeswalk['pop_jobs_ratio_1500_walk'] = (nodeswalk['pop_1500_walk'] / (nodeswalk['jobs_500_walk'])).fillna(0)
    nodeswalk['avg_hhs_500_walk'] = (nodeswalk['pop_500_walk'] / (nodeswalk['hh_500_walk'])).fillna(0)
    nodeswalk['avg_hhs_1500_walk'] = (nodeswalk['pop_1500_walk'] / (nodeswalk['hh_1500_walk'])).fillna(0)
    # end of addition
    
    # fill inf and nan with median
    
    def replace_inf_nan_with_median(col_name):
        return nodeswalk[col_name].replace([np.inf, -np.inf],np.nan).fillna(nodeswalk[col_name].median)
    
    for col_name in ['prop_children_500_walk','prop_singles_500_walk','prop_elderly_500_walk',
                     'prop_black_500_walk','prop_white_500_walk','prop_asian_500_walk','prop_hisp_500_walk',
                     'prop_rich_500_walk','prop_poor_500_walk','prop_children_1500_walk','prop_singles_1500_walk',
                     'prop_elderly_1500_walk','prop_black_1500_walk','prop_white_1500_walk','prop_asian_1500_walk',
                     'prop_hisp_1500_walk','prop_rich_1500_walk','prop_poor_1500_walk','pop_jobs_ratio_1500_walk',
                     'avg_hhs_500_walk','avg_hhs_1500_walk']:
        nodeswalk[col_name] = replace_inf_nan_with_median(col_name)
    
    
    print(nodeswalk.describe())
    orca.add_table('nodeswalk', nodeswalk)


@orca.step()
def network_aggregations_beam(netbeam):
    """
    This will be turned into a network aggregation template.
    """

    nodesbeam = networks.from_yaml(netbeam, 'network_aggregations_beam.yaml')
    nodesbeam = nodesbeam.fillna(0)
    print(nodesbeam.describe())
    orca.add_table('nodesbeam', nodesbeam)


@orca.injectable('output_parameters')
def output_parameters():
    with open("configs/output_parameters.yaml") as f:
        cfg = yaml.load(f)
        return cfg


def prepare_chart_data():

    output_parameters = orca.get_injectable('output_parameters')
    geo_small = output_parameters['chart_data']['geography_small']
    geo_large = output_parameters['chart_data']['geography_large']
    vars_sum = output_parameters['chart_data']['chart_variables']['sum']
    vars_mean = output_parameters['chart_data']['chart_variables']['mean']
    custom_variables = output_parameters['chart_data']['custom_charts']
    acres = orca.get_table('parcels').to_frame(['zone_id', 'acres'])\
        .groupby('zone_id').sum().reset_index()
    geo_attributes = orca.get_table('parcels').to_frame(
        ['zone_id', geo_small, geo_large]).\
        groupby('zone_id').min().reset_index()
    data = orca.get_table('zones').to_frame(vars_sum + vars_mean + ['zone_id'])
    data = pd.merge(data, acres, on='zone_id')
    data = pd.merge(data, geo_attributes, on='zone_id')
    for var in vars_sum:
        new_var_name = var + '_per_acre'
        data[new_var_name] = data[var] / data['acres']
        vars_sum = vars_sum + [new_var_name]
    variables = {'sum': vars_sum, 'mean': vars_mean}
    return data, variables, geo_small, geo_large, custom_variables

def aggregate_data(data, agg_type, geo):
    if (agg_type == 'mean'):
        data = data.groupby(geo).mean().reset_index()
    else:
        data = data.groupby(geo).sum().reset_index()
    return data

def gen_var_barcharts_by_geo(data, var, agg_type, geo):
    data = aggregate_data(data, agg_type, geo)
    titlex = agg_type + ' of ' + var.split('_', 1)[1].replace('_', ' ')
    titley = geo.replace('_', ' ')
    bar_chart = alt.Chart(data).mark_bar().encode(
        x = X(var, axis = Axis(title = titlex)),
        y = Y((geo + ':O'), axis = Axis(title=titley))
    )
    with open('./runs/%s_by_%s.json' % (var, geo), 'w') as outfile:
        json.dump(bar_chart.to_json(), outfile)

def gen_var_histograms(data, var, agg_type, geo, vdict, cdict):
    data = aggregate_data(data, agg_type, geo)
    data = data.copy()
    type = vdict[var].split(' ')[0]
    if type == 'Log':
        log_var = var
        data[log_var] = data[var]
    else:
        log_var = 'log_' + var
        data[log_var] = np.log(data[var])
    titlex = 'log of ' + var.split('_', 1)[1].replace('_', ' ')
    titley = 'number of '+ geo.replace('_', ' ') + 's'
    hist = alt.Chart(data).mark_bar().encode(
        alt.X(log_var, bin = True, axis=Axis(title = titlex)),
        alt.Y('count()', axis = Axis(title = titley))
    )
    with open('./runs/%s_histogram.json' % var, 'w') as outfile:
        json.dump(hist.to_json(), outfile)


def gen_var_scatters(data, var1, var2, agg1, agg2, geo_points, geo_large):
    colors = data.groupby(geo_points).min().reset_index()
    colors = colors[[geo_points, geo_large]]
    data_1 = aggregate_data(data, agg1, geo_points)[[var1 , geo_points]]
    data_2 = aggregate_data(data, agg2, geo_points)[[var2, geo_points]]
    data = pd.merge(data_1, data_2,on = geo_points, how = 'left' )
    data = pd.merge(data, colors, on = geo_points,how = 'left')
    titlex = agg1 +' of '+ var1.split('_', 1)[1].replace('_', ' ') + ' by zone'
    titley = agg2 +' of '+ var2.split('_', 1)[1].replace('_', ' ') + ' by zone'
    scatter = alt.Chart(data).mark_point().encode(
        x=X(var1, axis=Axis(title = titlex)),
        y=Y(var2, axis=Axis(title = titley)),
        color=geo_large + ':N',
    )
    with open('./runs/%s_vs_%s.json' % (var2, var1), 'w') as outfile:
        json.dump(scatter.to_json(), outfile)


def gen_barcharts_n_largest(data, var, agg_type, geo, n):
    data = aggregate_data(data, agg_type, geo)
    max_data = data.nlargest(n, var).reset_index()
    titlex = agg_type + ' of ' + var.split('_', 1)[1].replace('_', ' ')
    titley = geo.replace('_', ' ')
    bar_chart = alt.Chart(max_data).mark_bar().encode(
        x=X(var, axis=Axis(title = titlex)),
        y=Y(geo + ':O', axis=Axis(title=titley))
    )
    with open('./runs/%s_%ss_with_max_%s.json'% (n, geo, var), 'w') as outfile:
        json.dump(bar_chart.to_json(), outfile)


def gen_custom_barchart(table,var):
    df = orca.get_table(table).to_frame(['parcel_id', var]).\
        groupby(var).count().reset_index()
    df.rename(columns={'parcel_id': 'count_'+table}, inplace=True)
    chart = alt.Chart(df).mark_bar().encode(
        x=X('count_'+table, axis=Axis(title='count_'+table)),
        y=Y(var + ':O', axis=Axis(title=var))
    )
    with open('./runs/%s_by_%s.json'% (table, var), 'w') as outfile:
        json.dump(chart.to_json(), outfile)


def export_indicator_definitions():

    # Gets relevant data from output_parameters.yaml
    output_parameters = orca.get_injectable('output_parameters')
    indicator_vars = output_parameters['output_variables']
    sum_vars = output_parameters['chart_data']['chart_variables']['sum']
    mean_vars = output_parameters['chart_data']['chart_variables']['mean']
    geo_large = output_parameters['chart_data']['geography_large']
    geo_small = output_parameters['chart_data']['geography_small']
    custom_v = output_parameters['chart_data']['custom_charts']

    # Gets variable definitions from var_dict
    var_dict = orca.get_injectable('dictionary')['var_dict']
    custom_d = orca.get_injectable('dictionary')['custom_var_dict']

    # Creates dictionary with metadata for output indicators
    spatial_output = {}
    data = {}
    for geo_type in indicator_vars:
        desc = {}
        variables = indicator_vars[geo_type]
        for var in variables:
            geo_type = geo_type.strip('s')
            desc[var] = {'name': var_dict[var]}
        csv= geo_type + '_indicators'
        spatial_output[geo_type] = {'root_csv_name': csv,
                                                'var_display': desc}
    data['spatial_output'] = OrderedDict(spatial_output)

    # Creates dictionary with metadata for charts (based on parcel data)
    for var in sum_vars:
        new_var_name = var + '_per_acre'
        sum_vars = sum_vars + [new_var_name]
        var_dict[new_var_name] = var_dict[var] + ' per acre'
    desc = {}
    for var_type in ['sum', 'mean']:
        variables = eval(var_type+'_vars')
        for var in variables:
            name = ('% s_by_% s.json' % (var, geo_large))
            varname = var_dict[var].replace(': ',' ')
            desc[name] = {'title': 'By ' +geo_large+ ' code: ' + varname}
            name = ('%s_histogram.json' % var)
            varname = var_dict[var].strip('Log of ').replace(': ', ' ')
            desc[name] = {'title': 'Histogram: Logarithm of ' + varname.lower()}
            name = ('% s_% ss_with_max_% s.json'% (10, geo_small, var))
            varname = var_dict[var].replace(': ', ' ')
            desc[name] = {'title':'Top ten '
                                  + geo_small + ' codes: ' + varname}
        used_variables = []
        vars_chart = sum_vars + mean_vars
        for var1 in vars_chart:
            used_variables = used_variables + [var1]
            for var2 in vars_chart:
                if (var1 != var2) & (var2 not in used_variables):
                    name = ('%s_vs_%s.json' % (var2, var1))
                    varname1 = var_dict[var1].replace(': ', ' ')
                    varname2 = var_dict[var2].replace(': ', ' ')
                    desc[name] = {
                        'title': 'Zone Scatterplot: ' + varname2 +
                                 ' vs. ' + varname1}

    # Creates dictionary with metadata for charts (based on custom tables)
    for table in custom_v:
        for var in custom_v[table]:
            name = '%s_by_%s.json' % (table, var)
            key = table + '_' + var
            try:
                data_name = custom_d[key]['data_name']
                agg_name = custom_d[key]['aggregation_name']
            except Exception:
                data_name = 'Total ' + table.replace('_', ' ')
                agg_name =  var.replace('_', ' ')
            desc[name] = {'title': data_name +' by '+ agg_name}

    data['chart_output'] = OrderedDict(desc)

    # Exports dictionary with indicator and charts definitions to .yaml file
    data = OrderedDict(data)
    represent_dict_order = lambda self, data: \
        self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)
    yaml.Dumper.ignore_aliases = lambda *args: True
    with open('./runs/output_indicator_definitions.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, width = 1000)
    return var_dict, custom_d

@orca.step()
def generate_indicators(year, forecast_year, parcels, zones):
    # If iter_var is not defined is a presimulation generation
    if orca.get_injectable('iter_var'):
        year = orca.get_injectable('iter_var')
    else:
        year = orca.get_injectable('base_year')

    # General output indicators
    cfg = orca.get_injectable('output_parameters')['output_variables']
    zone_ind = zones.to_frame(cfg['zones'])
    zone_ind = zone_ind.reindex(sorted(zone_ind.columns),axis=1)
    parcel_ind = parcels.to_frame(cfg['parcels'])
    parcel_ind = parcel_ind.reindex(sorted(parcel_ind.columns),axis=1)
    zone_ind.to_csv('./runs/zone_indicators_%s.csv' % year)
    parcel_ind.to_csv('./runs/parcel_indicators_%s.csv' % year)

    # Output indicators by building type
    btype_columns = ['building_type_id','is_residential', 'is_non_residential']
    btypes = orca.get_table('building_types').to_frame(btype_columns)
    btypes = btypes.reset_index()
    btypes.loc[btypes['is_residential']==True, 'ind_res'] = \
        "sum_residential_units_" + btypes.building_type_id.astype(str)
    btypes.loc[btypes['is_non_residential'] == True, 'ind_non_res'] = \
        "sum_non_residential_sqft_" + btypes.building_type_id.astype(str)
    btype_ind_cols = list(btypes.ind_res) + list(btypes.ind_non_res)
    btype_ind_cols = [ind for ind in btype_ind_cols if str(ind) != 'nan']
    zone_type = zones.to_frame(btype_ind_cols)
    parcel_type = parcels.to_frame(btype_ind_cols)
    zone_type = zone_type.reindex(sorted(zone_type.columns), axis=1)
    parcel_type = parcel_type.reindex(sorted(parcel_type.columns), axis=1)
    zone_type.to_csv('./runs/zone_indicators_building_type_%s.csv' % year)
    parcel_type.to_csv('./runs/parcel_indicators_building_type_%s.csv' % year)

    # Generate chart indicators
    if year == forecast_year:
        vdict, cdict = export_indicator_definitions()
        data, variables, geo_small, geo_large, custom_v = prepare_chart_data()
        for table in custom_v:
            for var in custom_v[table]:
                gen_custom_barchart(table, var)
        used_variables = []
        for aggtype in ['sum', 'mean']:
            for var in variables[aggtype]:
                print ('Generating charts for ' + var)
                gen_var_barcharts_by_geo(data, var, aggtype, geo_large)
                gen_var_histograms(data, var, aggtype, geo_small, vdict, cdict)
                gen_barcharts_n_largest(data, var, aggtype, geo_small, 10)
                used_variables = used_variables + [var]
                for aggtype2 in ['sum', 'mean']:
                    for var2 in variables[aggtype2]:
                        if (var != var2) & (var2 not in used_variables):
                            gen_var_scatters(data, var, var2, aggtype,
                                             aggtype2, 'zone_id', geo_large)

#### Proforma ####

def parcel_average_price(use):
    cap_rate = 0.05  ## To convert from price/sqft to rent/sqft
    parcels = orca.get_table('site_proposals')
    if use == 'retail':
        price = parcels.pred_retail_price_ave_800_linear
    elif use == 'industrial':
        price = parcels.pred_industrial_price_ave_800_linear
    elif use == 'office':
        price = parcels.pred_office_price_ave_800_linear
    elif use == 'residential':
        mean_price = parcels.to_frame(['pred_sf_detached_price_ave_800_linear',
                                     'pred_duplex_townhome_price_ave_800_linear',
                                     'pred_multifamily_price_ave_800_linear']).mean(axis=1)
        mean_sqft_per_unit = orca.get_table('site_proposals').mean_sqft_per_unit_ave_800_linear
        mean_sqft_per_unit[mean_sqft_per_unit < 400] = 400
        price = mean_price / mean_sqft_per_unit

    price[price < 1] = 1
    return price * cap_rate


def parcel_is_allowed(form):
    """
    Defines which site proposals are allowed for a given form.
    Parameters
    ----------
    form : str
        The name of the form
    Returns
    -------
    A pandas series with "True" for the site proposals that are allowed
    under a given form, and "False" for the combinations that are not allowed.
    """
    id = 'parcel_zoning_id_combination'
    zoning = 'potential_zoning_id'
    btypes_df, btypes = gets_allowable_buildings(form)
    proposals = orca.get_table('site_proposals').to_frame([id, zoning])
    proposals.rename(columns={zoning: 'zoning_id'}, inplace=True)
    allowable = pd.merge(proposals, btypes_df,on = 'zoning_id',how='left')
    allowable = allowable[allowable['building_type_id'].isin(btypes)].copy()
    allowable = allowable[allowable['can_develop'] == True].copy()
    proposals['output'] = False
    proposals.loc[proposals[id].isin(allowable[id]), 'output'] = True

    return (proposals['output'])

def gets_allowable_buildings(form):
    """
    Helper function that gets the dataframe of allowable building types and
    matches the form with its allowable building types
    ----------
    form : str
        The name of the form
    Returns
    -------
    The DataFrame of allowable building types by zoning_id (including
    conditional costs by building type), and a list of the allowed building
    types for a given form
    """

    btypes = []
    if form =='industrial':
        btypes = [4400, 4900, 5100, 5200, 6300, 6400]
    elif form =='office':
        btypes = [4100,4300,4310,6600]
    elif form =='mix_non_residential':
        btypes = [2140]
    elif form == 'retail_office':
        btypes = [6110,6120,6130,6140,6150,6160,6170,6210,6500]
    elif form == 'retail':
        btypes = [4210,4220,4230,4240,4250,4260,4290,4320]
    elif form == 'mix_all':
        btypes = [9000]
    elif form == 'residential_office':
        btypes = [2121,2122,6220,6230,7200]
    elif form == 'residential_retail':
        btypes = [2111,2112,2131,2132,3100,3200,8000]
    elif form == 'residential':
        btypes = [1110,1121,1122,1130,1210,1220,7100,7300,7900]

    columns = ['zoning_id', 'building_type_id', 'can_develop',
               'conditional_use', 'cost_in_city', 'cost_in_ugb',
               'cost_outside_ugb', 'probability']
    allowable = orca.get_table('allowable_building_types').to_frame(columns)
    for i in ['cost_in_city', 'cost_in_ugb', 'cost_outside_ugb' ]:
        conditional = i +'_conditional'
        allowable.rename(columns={i: conditional}, inplace = True)
    return allowable, btypes

def parcel_custom_callback(parcels, pf):
    columns = ['developable_sqft', 'total_yearly_rent',
               'mean_sqft_per_unit_ave_800_linear']
    site_proposals = orca.get_table('site_proposals').to_frame(columns)
    parcels['parcel_size'] = site_proposals.developable_sqft
    parcels['land_cost'] = site_proposals.total_yearly_rent
    mean_sqft_per_unit = site_proposals.mean_sqft_per_unit_ave_800_linear
    mean_sqft_per_unit[mean_sqft_per_unit < 400] = 400
    parcels['ave_unit_size'] = mean_sqft_per_unit
    parcels = parcels[parcels.parcel_size > 2000]

    return parcels

def modifies_costs(self, form, newdf, total_development_costs):
    """
    Modifies total_development costs in two steps: 1) Adds zoning costs,
    conditional costs, and rezoning costs 2) Multiplies by cost shifters
    defined in configs/cost_shifters.yaml for calibration purposes.
    ----------
    form : str
        The name of the form.
    newdf: DataFrame
        Dataframe of allowed site proposals.
    total_development_costs: Array
        Array of costs before considering any planning-related costs,
        created by sqftproforma in the _lookup_parking_cfg function.
    Returns
    -------
    Array of total_development_costs including planning_costs and affected by
    cost shifters
    """

    costs = adds_planning_costs(self, form, newdf, total_development_costs)
    costs = cost_shifter_callback(self, form, newdf, costs)
    return costs


def adds_planning_costs(self, form, newdf, total_development_costs):
    """
    Adds zoning costs, conditional costs, and rezoning costs to
    total_development_costs, taking into account parcel location.
    ----------
    form : str
        The name of the form.
    newdf: DataFrame
        Dataframe of allowed site proposals.
    total_development_costs: Array
        Array of costs before considering any planning-related costs,
        created by sqftproforma in the _lookup_parking_cfg function.
    Returns
    -------
    Array of total_development_costs including planning_costs
    """

    newdf = planning_costs_by_status_ugb(form, newdf)
    costs = pd.DataFrame(total_development_costs).T
    planning_costs = pd.DataFrame(index=newdf.index,
                                  columns=costs.columns.tolist())

    for col in costs.columns.tolist():
        planning_costs[col] = newdf['planning_costs']
    costs = costs.add(planning_costs, fill_value=0).T.values

    return costs


def planning_costs_by_status_ugb(form, newdf):
    """
    Helper function that formats the dataframe of allowed site proposals,
    creating a new "planning_costs" column that accounts for any applicable
    zoning costs, conditional costs, or rezoning costs based on parcel location
    It also creates the zoning_btype table, which will be called later
    by the develop.py module to retrieve selected building types.
    ----------
    form : str
        The name of the form.
    newdf: DataFrame
        DataFrame of allowed site proposals.
    Returns
    -------
    Formatted DataFrame of allowed site proposals, including a new
    "planning_costs" column.

    """
    id = 'parcel_zoning_id_combination'
    newdf.rename(columns={'potential_zoning_id': 'zoning_id'}, inplace=True)
    btypes_df, btypes = gets_allowable_buildings(form)
    allowed_df = pd.merge(newdf, btypes_df, on='zoning_id', how='left')
    newdf.loc[newdf['rezoning_cost'] < -99999,'rezoning_cost'] = 0
    newdf.loc[newdf['rezoning_cost'].isnull(),'rezoning_cost'] = 0
    for i in ['in_city', 'in_ugb', 'outside_ugb']:
        cost = 'cost_' + i
        cond_cost = 'cost_' + i + '_conditional'
        bname = 'building_type_id_' + i
        status_ugb = 'with' + i
        if i == 'outside_ugb':
            status_ugb = i
        allowed_df.loc[allowed_df[cond_cost].isnull(), cond_cost] = 0
        allowed_df.loc[allowed_df[cond_cost] < -99999, cond_cost] = 0
        allowed_status = allowed_df.loc[allowed_df['status_ugb']==status_ugb]
        allowed_btype = selects_btype(allowed_status, cond_cost)
        allowed_btype.rename(columns={'building_type_id':bname}, inplace=True)
        newdf = pd.merge(newdf, allowed_btype,on=id,how='left')
        newdf.loc[newdf['status_ugb'] == status_ugb,'planning_costs'] = \
            newdf['rezoning_cost'] + newdf['overlay_cost'] + newdf[cond_cost]
        newdf.loc[newdf['status_ugb'] == status_ugb,'building_type_id'] = \
            newdf[bname]
        newdf = newdf.drop(columns=[cond_cost, bname])
    zoning_btype = newdf.copy()
    zoning_btype = zoning_btype[[id, 'building_type_id']]
    orca.add_table('zoning_btype_%s' %form, zoning_btype )

    return newdf

def selects_btype(df, cost):

    """
    Helper function that selects a building type for each zoning_id, based on
    the allowable_building_types table. If the scale factors for probabilities
    ('probability' column in the allowable_building_types table) are all equal
    to one for a given zoning_id, the first building type with the minimum
    conditional cost is selected. If the probability scale factors are
    different, the first building type with the maximum probability scale
    factor is selected.
    ----------
    df : DataFrame
        DataFrame of allowed site proposals, formatted by the
        planning_costs_by_status_ugb() function
    cost: str
        The name of the cost column
    Returns
    -------
    DataFrame with selected building_type_id for each allowed site proposal.

    """

    id = 'parcel_zoning_id_combination'
    df = df[[id, cost, 'building_type_id', 'probability']].copy()


    # Identifies method to assign btypes (min cost or max probability)
    pcount = df[[id,'probability']].groupby(id)['probability'].count().\
        reset_index().rename(columns={'probability': 'pcount'})
    psum = df[[id,'probability']].groupby(id)['probability'].sum().\
        reset_index().rename(columns={'probability': 'psum'})
    prob = pd.merge(pcount, psum, on=id, how='left')
    df = pd.merge(df, prob, on=id, how='left')
    df.loc[df['pcount'] != df['psum'], 'method'] = 'probability'
    df.loc[df['pcount'] == df['psum'], 'method'] = 'cost'

    # Creates dataframe with min conditional cost and max probability
    df_prob = df.loc[df['method']=='probability']
    df_cost = df.loc[df['method'] == 'cost']
    btypes_prob = df_prob.sort_values('probability').\
        groupby(id, as_index=False).last()
    btypes_cost = df_cost.sort_values(cost).groupby(id, as_index=False).first()
    btypes = btypes_prob.append(btypes_cost)
    btypes = btypes[[id, cost, 'building_type_id']]

    return btypes

@orca.injectable('cost_shifters')
def shifters():
    with open ("configs/cost_shifters.yaml") as f:
        cfg = yaml.load(f)
        return cfg

def cost_shifter_callback(self, form, df, costs):
    """
    Multiplies total_development costs (already including planning costs) by
    cost shifter values defined in cost_shifters.yaml by zone_district_id. This
    is done for calibration purposes
    ----------
    form : str
        The name of the form.
    df: DataFrame
        Dataframe of allowed site proposals.
    costs: Array
        Array of total_development costs, already considering planning-related
        costs.
    Returns
    -------
    Array of total_development_costs including planning_costs and multiplied by
    cost shifters
    """

    shifter_cfg = orca.get_injectable('cost_shifters')['calibration']
    geography = shifter_cfg['calibration_geography_id']
    shift_type = 'residential' if form == 'residential' else 'non_residential'
    shifters = shifter_cfg['proforma_cost_shifters'][shift_type]
    for geo, geo_df in df.reset_index().groupby(geography):
        shifter = shifters[geo]
        costs[:, geo_df.index] *= shifter
    return costs

def adds_btypes_proposals(feasibility):
    """
    Helper function that combines the individual zoning_btype tables into a
    single orca table for all forms (btypes_proposals). The btypes_proposals
    table allows retrieving the building type that was selected by the
    sqftproforma.py module for each parcel_zoning_id_combination. Note: These
    building types were selected using the planning_costs_by_status_ugb()
    function, which selects the building type with minimum conditional costs
    or maximum probability for each zoning_id. Conditional costs and
    probabilities are defined by the user in the allowable_building_types.csv
    file.

    ----------
    feasibility : DataFrame
        Table with the results from the feasibility step (sqftproforma.py)

    Returns
    -------
    None. Registers btypes_proposals table with Orca.

    """
    id = 'parcel_zoning_id_combination'
    df = feasibility.copy()
    with open("./configs/proforma.yaml") as f:
        cfg = yaml.load(f)
    forms = cfg['forms_to_test']
    for form in forms:
        btypes_cols = [id, 'building_type_id']
        btypes = orca.get_table('zoning_btype_%s' %form).to_frame(btypes_cols)
        btypes.rename(columns={'building_type_id': 'btype_form'}, inplace=True)
        df = pd.merge(df, btypes, on=id, how='left')
        df.loc[df['form'] == form, 'building_type_id'] = df['btype_form']
        df = df.drop(columns=['btype_form'])
    df = df[[id,'building_type_id']].copy()
    df = df.groupby(id).min().reset_index()
    orca.add_table('btypes_proposals', df)

def formats_feasibility(site_proposals):
    """
    Adds desired columns from site_proposals into the feasibility table

    """
    id = 'parcel_zoning_id_combination'
    feasibility = orca.get_table('feasibility').to_frame()
    feasibility['parcel_id'] = site_proposals.parcel_id
    feasibility[id] = site_proposals[id]
    feasibility['original_zoning_id'] = site_proposals.zoning_id
    feasibility['zoning_id'] = site_proposals.potential_zoning_id
    feasibility['overlay_id'] = site_proposals.overlay_id
    feasibility['annexed_overlay_id'] = site_proposals.annexed_overlay_id
    feasibility['city'] = site_proposals.city
    feasibility['ugb'] = site_proposals.ugb
    feasibility = feasibility.set_index('parcel_id')
    orca.add_table('feasibility', feasibility)
    adds_btypes_proposals(feasibility)

def scales_probability(df):
    """
    Helper function passed as 'profit_to_prob_func' to the pick() method in the
    develop.py module. This function first retrieves building type that was
    used by the sqftproforma.py module to calculate conditional costs for a
    given zoning_id. Then, this previously selected building type is used to
    identify the corresponding user_defined probability scale factor (from the
    'allowable_building_types' table). The scale factor is applied to calculate
    the final probability of choosing a given building within the develop.py
    module.

    ----------
    df : DataFrame
        DataFrame of potential buildings from SqFtProForma steps, formatted by
        the pick() method in the develop.py module
    Returns
    -------
    DataFrame with probabilities for each record (building) in the input
    dataframe.

    """

    id = 'parcel_zoning_id_combination'
    df = df[[id,'zoning_id','max_profit','parcel_size']]
    btypes = orca.get_table('btypes_proposals').to_frame()
    df = pd.merge(df, btypes, on=id, how='left')
    df['zoning_building'] = df['zoning_id'].astype(int).astype(str) +'_'+ \
                            df['building_type_id'].astype(int).astype(str)
    p_factor = orca.get_table('allowable_building_types').to_frame \
        (['zoning_id', 'building_type_id', 'probability'])
    p_factor['zoning_building'] = p_factor['zoning_id'].astype(str)+'_'+\
                                  p_factor['building_type_id'].astype(str)
    p_factor = p_factor.drop(columns=['zoning_id', 'building_type_id'])
    df = pd.merge(df.reset_index(), p_factor, on='zoning_building')\
        .set_index('index')
    df = df.drop(columns=['zoning_building'])
    df['max_profit_per_size'] = (df.max_profit / df.parcel_size)
    df['scaled_max_profit_per_size'] = df.probability *(df.max_profit_per_size)
    appended_probabilities = pd.DataFrame()
    for btype in df.building_type_id.unique():
        df_btype = df[df['building_type_id']==btype].copy()
        total_profit_btype = df_btype.scaled_max_profit_per_size.sum()
        df_btype['probabilities'] = df_btype.scaled_max_profit_per_size\
                                    /(total_profit_btype)
        appended_probabilities = appended_probabilities.append(df_btype)
    return appended_probabilities

def update_annexed_col(parcelsdf):
    @orca.column('parcels', 'annexed', cache=True, cache_scope='step')
    def func():
        series = pd.Series(data = parcelsdf.annexed, index =parcelsdf.index)
        return series
def update_city(parcelsdf):
    @orca.column('parcels', 'city', cache=True, cache_scope='step')
    def func():
        series = pd.Series(data = parcelsdf.city, index =parcelsdf.index)
        return series
    return func
def update_overlay_id(parcelsdf):
    @orca.column('parcels', 'overlay_id', cache=True, cache_scope='step')
    def func():
        series = pd.Series(data = parcelsdf.overlay_id, index = parcelsdf.index)
        return series
    return func
def update_zoning_cols(parcelsdf, col):
    @orca.column('parcels', col, cache=True, cache_scope='step')
    def func():
        series = pd.Series(data=parcelsdf[col], index=parcelsdf.index)
        return series
    return func

def update_annexed(new_buildings):
    """
     Updates the 'city' and 'overlay_id fields for parcels that get annexed
     during the simulation year. Prints number of developed, rezoned, and
     annexed parcels.
     ----------
     new_buildings: DataFrame
         Table with the buildings that were selected by the developer model

     Returns
     -------
     None
     """
    new_buildings['rezoned'] = 0
    new_buildings.loc[new_buildings.zoning_id !=
                      new_buildings.original_zoning_id, 'rezoned'] = 1
    new_buildings = new_buildings.copy().sort_values('rezoned').\
        groupby('parcel_id', as_index=False).last()
    parcel_cols = ['parcel_id', 'city', 'ugb', 'overlay_id']
    parcels = orca.get_table('parcels').to_frame(parcel_cols).reset_index()
    parcels['developed'] = 0
    parcels['annexed'] = 0
    parcels.loc[parcels.parcel_id.isin(new_buildings.parcel_id),
                'developed'] = 1
    parcels.loc[(parcels['developed'] == 1) & (parcels['ugb'].notnull()) &
                (parcels['city'].isnull()),'annexed'] = 1
    parcels.loc[parcels.annexed == 1, 'city'] = parcels.ugb
    overlays = orca.get_table('zone_overlay_types').to_frame()
    overlays = overlays[overlays['overlay_id']
                        != overlays['annexed_overlay_id']].copy()
    cols = overlays.columns.drop(['overlay_id', 'annexed_overlay_id',
                                  'overlay_combination', 'cost_in_city',
                                  'cost_in_ugb', 'cost_outside_ugb'])
    for col in cols:
        overlays = overlays.rename(columns={col: col + '_overlay'})
    parcels = parcels.\
        merge(overlays, on='overlay_id',how='left').set_index('parcel_id')
    parcels.loc[
        parcels.annexed == 1, 'overlay_id'] = parcels.annexed_overlay_id
    annexed = parcels[parcels.annexed==1].copy().\
        groupby('city', as_index=False).sum()
    update_annexed_col(parcels)
    update_city(parcels)
    update_overlay_id(parcels)
    for col in cols:
        col_overlay = col + '_overlay'
        parcels.loc[(parcels.annexed==1) & (parcels[col_overlay].notnull()),
                    col] = parcels[col_overlay]
        update_zoning_cols(parcels, col)
    print ('Total parcels that will develop: ',
           new_buildings.parcel_id.nunique())
    print ('Total rezoned parcels: ', new_buildings.rezoned.sum())
    for city in annexed.city.unique():
        print ('Total annexed parcels: ',city,': ',
               annexed[annexed['city']==city].annexed.item())

def add_extra_columns(df):
    df['units'] = df.residential_units + df.non_residential_sqft
    for col in ['maplot', 'improvement_value', 'imputed']:
        df[col] = 0
    df['impval_per_unit'] = 1 # Placeholder
    btypes_columns = ['parcel_zoning_id_combination', 'building_type_id']
    btypes = orca.get_table('btypes_proposals').to_frame(btypes_columns)
    df = pd.merge(df, btypes, on='parcel_zoning_id_combination', how='left')
    update_annexed(df)
    return df

def custom_selection(self, df, p, target_units):
    btypes = orca.get_table('btypes_proposals').to_frame()
    df = pd.merge(df, btypes, on='parcel_zoning_id_combination', how ='left')
    selected = np.array([])
    for btype in target_units.index.get_values():
        target_units_btype = target_units.loc[btype].get_value('target_units')
        df_btype = df[df['building_type_id']==btype]
        p_btype = p[p.building_type_id == btype].probabilities
        sample_size = int(min(len(df_btype.index), target_units_btype))
        if sample_size != 0:
            choices_btype =  proposal_select.\
                weighted_random_choice_multiparcel\
                (df_btype, p_btype, target_units_btype)
            selected = np.append(selected, choices_btype)
    return selected


@orca.step()
def feasibility(site_proposals):
    parcel_utils.run_feasibility(site_proposals,
                                 parcel_average_price,
                                 parcel_is_allowed,
                                 cfg='proforma.yaml',
                                 parcel_custom_callback=parcel_custom_callback,
                                 modify_costs=modifies_costs)
    formats_feasibility(site_proposals)


@orca.step()
def residential_developer(feasibility, households, buildings, parcels, year):
    target_vacancies = orca.get_table('target_vacancies').to_frame()
    new_buildings = parcel_utils.run_developer(
            "residential",
            households,
            buildings,
            'residential_units',
            feasibility,
            parcels.developable_sqft,
            parcels.mean_sqft_per_unit,
            parcels.sum_residential_units,
            'res_developer.yaml',
            year=year,
            target_vacancy = target_vacancies.reset_index(),
            form_to_btype_callback=None,
            add_more_columns_callback=add_extra_columns,
            profit_to_prob_func=scales_probability,
            custom_selection_func=custom_selection)


@orca.step()
def non_residential_developer(feasibility, jobs, buildings, parcels, year):
    target_vacancies = orca.get_table('target_vacancies').to_frame()
    new_buildings = parcel_utils.run_developer(
            ["office", "retail", "industrial"],
            jobs,
            buildings,
            'job_spaces',
            feasibility,
            parcels.developable_sqft,
            parcels.mean_sqft_per_unit,
            parcels.sum_job_spaces,
            'nonres_developer.yaml',
            year=year,
            target_vacancy=target_vacancies.reset_index(),
            form_to_btype_callback=None,
            add_more_columns_callback=add_extra_columns,
            profit_to_prob_func=scales_probability,
            custom_selection_func=custom_selection)


#### Transition

def full_transition(agents, agent_controls, totals_column, year,
                    location_fname, linked_tables=None,
                    accounting_column=None, set_year_built=False):
    """
    Run a transition model based on control totals specified in the usual
    UrbanSim way
    Parameters
    ----------
    agents : DataFrameWrapper
        Table to be transitioned
    agent_controls : DataFrameWrapper
        Table of control totals
    totals_column : str
        String indicating the agent_controls column to use for totals.
    year : int
        The year, which will index into the controls
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)
    linked_tables : dict, optional
        Sets the tables linked to new or removed agents to be updated with
        dict of {'table_name':(DataFrameWrapper, 'link_id')}
    accounting_column : str, optional
        Name of column with accounting totals/quantities to apply toward the
        control. If not provided then row counts will be used for accounting.
    set_year_built: boolean
        Indicates whether to update 'year_built' columns with current
        simulation year
    Returns
    -------
    Nothing
    """
    ct = agent_controls.to_frame()
    agnt = agents.local
    print("Total agents before transition: {}".format(len(agnt)))
    tran = transition.TabularTotalsTransition(ct, totals_column,
                                              accounting_column)
    updated, added, copied, removed = tran.transition(agnt, year)
    updated.loc[added, location_fname] = -1
    if set_year_built:
        updated.loc[added, 'year_built'] = year

    updated_links = {}
    if linked_tables:
        for table_name, (table, col) in linked_tables.iteritems():
            print('updating linked table {}'.format(table_name))
            updated_links[table_name] = \
                update_linked_table(table, col, added, copied, removed)
            orca.add_table(table_name, updated_links[table_name])

    print("Total agents after transition: {}".format(len(updated)))
    orca.add_table(agents.name, updated[agents.local_columns])
    return updated, added, copied, removed


@orca.step('household_transition')
def household_transition(households, annual_household_control_totals, year):
    full_transition(households, annual_household_control_totals,
                           'total_number_of_households', year, 'building_id')


@orca.step('job_transition')
def job_transition(jobs, annual_employment_control_totals, year):
    full_transition(jobs, annual_employment_control_totals,
                            'total_number_of_jobs', year, 'building_id')

