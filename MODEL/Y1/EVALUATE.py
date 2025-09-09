import os
import time
import yaml
import argparse


def main(tag, index, folder):
    '''
    Main function to create the model evaluator configuration file.
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the function in minutes
    '''
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    
    # Config
    config = {
        'EVALUATE{}'.format(index): {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'model': 'input_model',
                'output': 'output_data'
            }, 
            'chunk_size': 100000, 
            'nondetect_val': 99.0, 
            'ref_band': 'mag_i_lsst', 
            'output_mode': 'default',
            'qp_representation': 'interp',
            'hdf5_groupname': 'photometry', 
            'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301, 
            'bands': [
                'mag_u_lsst', 
                'mag_g_lsst', 
                'mag_r_lsst', 
                'mag_i_lsst', 
                'mag_z_lsst', 
                'mag_y_lsst'
            ], 
            'err_bands': [
                'mag_u_lsst_err', 
                'mag_g_lsst_err', 
                'mag_r_lsst_err', 
                'mag_i_lsst_err', 
                'mag_z_lsst_err', 
                'mag_y_lsst_err'
            ]
        }
    }
    
    os.makedirs(os.path.join(model_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(model_folder, '{}/EVALUATE/'.format(tag)), exist_ok=True)
    
    config_name = os.path.join(model_folder, '{}/EVALUATE/EVALUATE{}.yaml'.format(tag, index))
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
    PARSE = argparse.ArgumentParser(description='Model Evaluator')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)