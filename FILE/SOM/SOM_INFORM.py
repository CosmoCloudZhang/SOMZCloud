import os
import yaml
import argparse

def main(path):
    
    # Path
    data_path = os.path.join(path, 'DATA/')
    
    # Config
    config = {
        'SOM_INFORM': {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'model': 'input_model',
            }, 
            'seed': 0, 
            'som_sigma': 0.5, 
            'nondetect_val': 99.0, 
            'ref_band': 'mag_i_lsst', 
            'som_learning_rate': 0.60, 
            'm_dim': 125, 'n_dim': 125, 
            'som_iterations': 10000000,
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
                'mag_err_u_lsst', 
                'mag_err_g_lsst', 
                'mag_err_r_lsst', 
                'mag_err_i_lsst', 
                'mag_err_z_lsst', 
                'mag_err_y_lsst'
            ]
        }
    }
    
    config_name = os.path.join(data_path, 'SOM/SOM_INFORM.yaml')
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='SOM Informer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    
    PATH = PARSE.parse_args().path
    main(PATH)