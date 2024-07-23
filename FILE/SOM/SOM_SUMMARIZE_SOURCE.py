import os
import yaml
import argparse

def main(path, index):
    
    # Path
    data_path = os.path.join(path, 'DATA/')
    
    # Config
    config = {
        'SOM_SUMMARIZE_SOURCE{}'.format(index): {
            'aliases': {
                'name': 'input_name', 
                'input': 'input_data', 
                'model': 'input_model',
                'output': 'output_data',
                'spec_input': 'input_spec',
                'single_NZ': 'output_single', 
                'cellid_output': 'output_cellid',
                'uncovered_cluster_file': 'output_cluster',
            }, 
            'seed': 0,
            'split': 10000, 
            'nsamples': 1000, 
            'single_NZ': None,
            'spec_input': None, 
            'chunk_size': 2000000, 
            'nondetect_val': 99.0, 
            'redshift_colname': 'redshift', 
            'hdf5_groupname': 'photometry', 
            'spec_groupname': 'photometry', 
            'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301
        }
    }
    
    config_name = os.path.join(data_path, 'SOM/SOM_SUMMARIZE_SOURCE{}.yaml'.format(index))
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='SOM Summarizer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--index', type=int, required=True, help='Index of the sample for the modelling')
    
    PATH = PARSE.parse_args().path
    INDEX = PARSE.parse_args().index
    
    main(PATH, INDEX)