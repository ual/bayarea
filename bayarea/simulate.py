import sys
import time
import argparse
import numpy as np

import orca

from lcog import datasources
from lcog import variables
from lcog import models

### Template imports
from urbansim.models import util
from urbansim_templates import modelmanager as mm
from urbansim_templates.models import OLSRegressionStep

mm.initialize()


def run(forecast_year=2035, random_seed=False):
    """
    Set up and run simulation.
    Parameters
    ----------
    forecast_year : int, optional
        Year to simulate to. If year argument is passed from the terminal, then
        that year is applied here, otherwise a default value is applied.
    random_seed : int, optional
        Random seed.
    Returns
    -------
    _ : None
        No return value for now.
    """
    # Record start time
    start_time = time.time()

    orca.add_injectable('forecast_year', forecast_year)

    # Set value of optional random seed
    if random_seed:
        np.random.seed(random_seed)

    # Model names
    transition_models = ['household_transition', 'job_transition']
    price_models = ['repm_sf_detached', 'repm_duplex_townhome', 'repm_multifamily',
                    'repm_industrial', 'repm_retail', 'repm_office']
    developer_models = ['feasibility', 'residential_developer', 'non_residential_developer']
    location_models = ['hlcm1', 'hlcm2',
                       'elcm1', 'elcm2', 'elcm3', 'elcm4', 'elcm5', 'elcm6',
                       'elcm7', 'elcm8', 'elcm9', 'elcm10', 'elcm11', 'elcm12',
                       'elcm13', 'elcm14']
    end_of_year_models = ['generate_indicators']

    # Simulate
    orca.run(['build_networks', 'generate_indicators'])
    orca.run(transition_models + price_models + developer_models + location_models + end_of_year_models,
             iter_vars = list(range(2011, forecast_year + 1)))


    # Record end time
    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Simulation duration: %s minutes' % (time_elapsed/60))


if __name__ == '__main__':

    # Run simulation with optional command-line arguments
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-y", "--year", type=int, help="forecast year to simulate to")
        parser.add_argument("-s", "--seed", type=float, help="random seed value")
        args = parser.parse_args()

        forecast_year = args.year if args.year else 2035
        random_seed = int(args.seed) if args.seed else False
        run(forecast_year, random_seed)

    else:
        run()
