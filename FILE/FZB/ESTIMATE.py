import os
import yaml
import argparse

def main(index, folder):
    """
    Main function to create the INFORM.yaml file
    
    Arguments:
        index (int): Index of the dataset
        folder (str): The base folder of the datasets
    
    Returns:
        None
    """
    # Path
    data_folder = os.path.join(folder, 'FZB')
    
    # Config
    config = {
        'ESTIMATE{}'.format(index): {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'model': 'input_model',
                'output': 'output_data'
            }, 
            'chunk_size': 1000000, 
            'nondetect_val': 30.0, 
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
                'mag_err_u_lsst', 
                'mag_err_g_lsst', 
                'mag_err_r_lsst', 
                'mag_err_i_lsst', 
                'mag_err_z_lsst', 
                'mag_err_y_lsst'
            ]
        }
    }
    
    config_name = os.path.join(data_folder, 'ESTIMATE{}.yaml'.format(index))
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