import pytest
import orca

from lcog import datasources


@pytest.fixture
def expected_injectables():
    expected_injectables = ['store', 'aggregate_geos', 'year', 'dictionary']
    return expected_injectables


@pytest.fixture
def expected_tables():
    expected_tables = ['parcels',
                       'buildings',
                       'jobs',
                       'households',
                       'households_pums',
                       'travel_data',
                       'nodes',
                       'edges',
                       'annual_employment_control_totals',
                       'annual_household_control_totals',
                       'zonings',
                       'locations',
                       'block_groups',
                       'blocks',
                       'zones',
                       'plans',
                       'zone_districts',
                       'zone_subdistricts',
                       'plan_types',
                       'zone_types',
                       'plan_compatible_zones',
                       'building_types',
                       'allowable_building_types',
                       'building_sqft_per_job',
                       'site_proposals',
                       'target_vacancies']
    return expected_tables


def test_injectable_list(expected_injectables):
    print(orca.list_injectables())
    assert orca.list_injectables() == expected_injectables


def test_table_list(expected_tables):
    print(orca.list_tables())
    assert orca.list_tables() == expected_tables
