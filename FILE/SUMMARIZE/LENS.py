import os
import yaml
import time
import argparse

def main(path, index):
    
    # Path
    data_path = os.path.join(path, 'DATA/')
    
    # Config
    config = {
        'LENS_SUMMARIZE{}'.format(index): {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'output': 'output_data', 
                'single_NZ': 'single_NZ'
            }, 
            'seed': 0,
            'name': None,
            'input': None,
            'output': None,
            'config': None,
            'nsamples': 250,
            'single_NZ': None,
            'chunk_size': 2000000, 
            'zmin': 0.0, 'zmax': 3.0, 'nzbins': 300,
        }
    }
    
    config_name = os.path.join(data_path, 'LENS/LENS{}/CONFIG.yaml'.format(index))
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Estimator')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--index', type=int, required=True, help='Index of the sample for the modelling')
    
    PATH = PARSE.parse_args().path
    INDEX = PARSE.parse_args().index
    print('Index: {}'.format(INDEX))
    
    START = time.time()
    main(PATH, INDEX)
        
    END = time.time()
    print('Time: {:.2f} minutes'.format((END - START) / 60))