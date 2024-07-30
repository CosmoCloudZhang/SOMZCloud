import os
import yaml
import argparse

def main(path, index):
    
    # Path
    data_path = os.path.join(path, 'DATA/')
    
    # Config
    config = {
        'FZB_INFORM{}'.format(index): {
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
            'max_basis': 100, 'basis_system': 'cosine', 
            'bumpmin': 0.0, 'bumpmax': 0.5, 'nbump': 50, 
            'sharpmin': 0.5, 'sharpmax': 2.5, 'nsharp': 50, 
            'regression_params': {
                'verbosity': 0, 
                'max_depth': 16, 
                'learning_rate': 0.02,
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
    
    config_name = os.path.join(data_path, 'FZB/FZB_INFORM{}.yaml'.format(index))
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Informer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the train datasets')
    
    PATH = PARSE.parse_args().path
    INDEX = PARSE.parse_args().index
    
    main(PATH, INDEX)
    print('Index: {}'.format(INDEX))