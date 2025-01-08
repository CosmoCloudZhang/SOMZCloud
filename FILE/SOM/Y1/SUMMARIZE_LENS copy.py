import os
import time
import yaml
import argparse

def main(bin, index, folder):
    '''
    Create the configuration file for the SOM Summarizer.
    
    Arguments:
        bin (int): The tomographic bin.
        index (int): Index of the datasets.
        folder (str): The path to the base folder.
    '''
    # Start
    start = time.time()
    print('Bin: {} Index: {}'.format(bin, index))
    
    # Path
    som_folder = os.path.join(folder, 'SOM/')
    os.makedirs(os.path.join(som_folder, 'LENS'), exist_ok=True)
    os.makedirs(os.path.join(som_folder, 'LENS/LENS{}'.format(index)), exist_ok=True)
    
    # Config
    config = {
        'SUMMARIZE_LENS{}'.format(index): {
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
            'split': 10000, 
            'nsamples': 1000, 
            'n_clusters': 10000, 
            'chunk_size': 50000, 
            'nondetect_val': 99.0, 
            'redshift_colname': 'redshift', 
            'hdf5_groupname': 'photometry', 
            'spec_groupname': 'photometry', 
            'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301
        }
    }
    
    config_name = os.path.join(som_folder, 'LENS/LENS{}/SUMMARIZE{}.yaml'.format(index, bin))
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)
    
    # End
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='SOM Summarizer')
    PARSE.add_argument('--bin', type=int, required=True, help='The tomographic bin')
    PARSE.add_argument('--index', type=int, help='The index of the dataset')
    PARSE.add_argument('--folder', type=str, help='The path to the base folder')
    
    # Parse
    BIN = PARSE.parse_args().bin
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(BIN, INDEX, FOLDER)