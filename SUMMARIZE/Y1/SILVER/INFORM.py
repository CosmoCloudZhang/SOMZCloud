import os
import time
import yaml
import argparse


def main(tag, name, index, folder):
    '''
    Generate configuration file for the SOMoclu Informer.
    
    Arguments:
        tag (int) : the tag of configuration
        name (str) : the name of configuration
        index (int) : the index of all the datasets
        folder (str) : the base folder of all the datasets
    
    Returns:
        duration (float) : the duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    os.makedirs(os.path.join(summarize_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(summarize_folder, '{}/{}/INFORM/'.format(tag, name)), exist_ok=True)
    
    # Config
    config = {
        'INFORM{}'.format(index): {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'model': 'input_model',
            }, 
            'n_rows': 100, 
            'n_epochs': 100, 
            'n_columns': 100, 
            'std_coeff': 0.5, 
            'maptype': 'toroid', 
            'nondetect_val': 99.0, 
            'grid_type': 'hexagonal', 
            'ref_band': 'mag_i_lsst',
            'som_learning_rate': 0.1, 
            'column_usage': 'magandcolors',
            'hdf5_groupname': 'photometry', 
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
    
    # Save
    config_name = os.path.join(summarize_folder, '{}/{}/INFORM/INFORM{}.yaml'.format(tag, name, index))
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Summarize Silver Informer')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, INDEX, FOLDER)