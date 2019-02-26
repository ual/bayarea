import os
from pathlib import PurePath
import importlib
from lcog.tests.config import config

print('Processing and running {} jupyter '
      'notebook files: {}...'.format(len(config.TEST_NB_LIST),
                                     config.TEST_NB_LIST))
for jupyter_nb_path in config.TEST_NB_LIST:
    # TODO: jupyter_nb file name must not have any spaces, if they do process
    #  will fail. Consider adding replace whitespace with underscores to
    #  support spaces in file names
    nb_file_name = PurePath(jupyter_nb_path).parts[-1]
    if nb_file_name not in config.SKIP_NB:
        config.convert_nb_to_py(input_file=jupyter_nb_path,
                                output_dir=config.PY_OUTPUT_DIR)
        input_file_name = nb_file_name.replace('.ipynb', '.py')
        input_file = os.path.join(config.PY_OUTPUT_DIR, input_file_name)
        output_file_name = config.remove_magic_lines(input_file=input_file)
        py_file_to_run = output_file_name.replace('.py', '')
        module_name = 'lcog.{}'.format(py_file_to_run)
        print('----- Running file: {} -----'.format(output_file_name))
        importlib.import_module(module_name, package=None)
        print('----- Completed running file: {} -----'.format(
            output_file_name))
print('Completed processing and running jupyter notebook files')
