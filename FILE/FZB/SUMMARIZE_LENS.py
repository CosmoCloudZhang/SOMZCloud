import os
import time
import yaml
import argparse


def main(bin, index, folder):
    '''
    Main function to create the FZB informer configuration file.
    
    Arguments:
        index (int): The index of the dataset.
        folder (str): The base folder of the datasets.
    
    Returns:
        duration (float): The duration of the function in minutes.
    '''
    # Start
    start = time.time()
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    
    # Config
    config = {
        'SUMMARIZE{}'.format(index): {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'output': 'output_data', 
                'single_NZ': 'single_data'
            }, 
            'seed': 0, 
            'nsamples': 1000, 
            'chunk_size': 400000, 
            'zmin': 0.0, 'zmax': 3.0, 'nzbins': 300
        }
    }
    
    os.makedirs(os.path.join(fzb_folder, 'LENS/LENS{}'.format(index)), exist_ok=True)
    config_name = os.path.join(fzb_folder, 'LENS/LENS{}/SUMMARIZE{}.yaml'.format(index, bin))
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)
    
    # End
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Index: {} Time: {:.2f} minutes'.format(index, duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Summarizer')
    PARSE.add_argument('--bin', type=str, required=True, help='The tomographic bin')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    BIN = PARSE.parse_args().bin
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(BIN, INDEX, FOLDER)