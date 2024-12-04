import os
import yaml
import argparse


def main(index, folder):
    '''
    Main function to create the SOM informer configuration file.
    
    Arguments:
        index (int): The index of the dataset.
        folder (str): The base folder of the datasets.
    
    Returns:
        duration (float): The duration of the function in minutes.
    '''
    # Path
    som_folder = os.path.join(folder, 'SOM/')
    
    # Config
    config = {
        'INFORM{}'.format(index): {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'model': 'input_model',
            }, 
            'seed': 0, 
            'std_coeff': 0.5, 
            'maptype': 'toroid', 
            'nondetect_val': 30.0, 
            'grid_type': 'hexagonal', 
            'ref_band': 'mag_i_lsst',
            'column_usage': 'colors',
            'som_learning_rate': 0.5, 
            'hdf5_groupname': 'photometry', 
            'n_rows': 100, 'n_columns': 100, 
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
    
    os.makedirs(os.path.join(som_folder, 'INFORM'), exist_ok=True)
    config_name = os.path.join(som_folder, 'INFORM/INFORM{}.yaml'.format(index))
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)


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