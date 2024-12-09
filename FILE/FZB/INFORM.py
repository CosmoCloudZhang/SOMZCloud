import os
import time
import yaml
import argparse


def main(index, folder):
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
        'INFORM{}'.format(index): {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'model': 'input_model',
            }, 
            'save_train': True, 
            'nondetect_val': 99.0, 
            'output_mode': 'default',
            'ref_band': 'mag_i_lsst', 
            'redshift_col': 'redshift', 
            'hdf5_groupname': 'photometry', 
            'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301, 
            'trainfrac': 0.75, 'retrain_full': True,
            'max_basis': 50, 'basis_system': 'Fourier', 
            'bumpmin': 0.0, 'bumpmax': 0.5, 'nbump': 50, 
            'sharpmin': 0.5, 'sharpmax': 2.5, 'nsharp': 50, 
            'regression_params': {
                'max_depth': 8, 
                'learning_rate': 0.1,
                'objective': 'reg:squarederror'
            }, 
            'bands': [
                'mag_u_lsst', 
                'mag_g_lsst', 
                'mag_r_lsst', 
                'mag_i_lsst', 
                'mag_z_lsst', 
                'mag_y_lsst'
            ], 
            'err_bands': [
                'mag_err_u_lsst', 
                'mag_err_g_lsst', 
                'mag_err_r_lsst', 
                'mag_err_i_lsst', 
                'mag_err_z_lsst', 
                'mag_err_y_lsst'
            ]
        }
    }
    
    os.makedirs(os.path.join(fzb_folder, 'INFORM'), exist_ok=True)
    config_name = os.path.join(fzb_folder, 'INFORM/INFORM{}.yaml'.format(index))
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
    PARSE = argparse.ArgumentParser(description='FZB Informer')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(INDEX, FOLDER)