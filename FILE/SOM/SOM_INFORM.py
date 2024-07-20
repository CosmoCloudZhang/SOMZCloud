import os
import yaml
import argparse

def main(path, index):
    
    # Path
    data_path = os.path.join(path, 'DATA/')
    
    # Config
    config = {
        'SOM_INFORM{}'.format(index): {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'model': 'input_model',
            }, 
            'seed': 0, 
            'name': None,
            'input': None, 
            'model': None,
            'config': None, 
            'std_coeff': 10.0, 
            'nondetect_val': 99.0, 
            'ref_band': 'mag_i_lsst', 
            'output_mode': 'default', 
            'som_learning_rate': 0.005,
            'redshift_col': 'redshift', 
            'n_rows': 50, 'n_columns': 50, 
            'hdf5_groupname': 'photometry', 
            'column_usage': 'magandcolors', 
            'maptype': 'planar', 'grid_type': 'hexagonal', 
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
    
    config_name = os.path.join(data_path, 'SOM/SOM_INFORM{}.yaml'.format(index))
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='SOM Informer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the train datasets')
    
    PATH = PARSE.parse_args().path
    INDEX = PARSE.parse_args().index
    
    main(PATH, INDEX)
    print('Index: {}'.format(INDEX))