import os
import yaml
import argparse

def main(path, index):
    
    # Path
    data_path = os.path.join(path, 'DATA/')
    os.makedirs(os.path.join(data_path, 'FZB/LENS'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'FZB/LENS/LENS{}'.format(index)), exist_ok=True)
    
    # Config
    config = {
        'FZB_SUMMARIZE_LENS{}'.format(index): {
            'aliases': {
                'name': 'input_name', 
                'input': 'input_data', 
                'output': 'output_data',
                'single_NZ': 'output_single', 
            }, 
            'seed': 0,
            'nsamples': 1000, 
            'chunk_size': 10000, 
            'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301
        }
    }
    
    config_name = os.path.join(data_path, 'FZB/LENS/LENS{}/FZB_SUMMARIZE.yaml'.format(index))
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Summarizer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--index', type=int, required=True, help='Index of the sample for the modelling')
    
    PATH = PARSE.parse_args().path
    INDEX = PARSE.parse_args().index
    
    main(PATH, INDEX)
    print('INDEX: {}'.format(INDEX))