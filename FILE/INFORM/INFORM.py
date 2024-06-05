import os
import yaml
import time
import argparse

def main(path, index):
    
    # Path
    file_path = os.path.join(path, 'FILE/')
    
    # Config
    config = {
        'FZB_INFORM{}'.format(index): {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'model': 'input_model',
            }, 
            'name': None,
            'input': None, 
            'model': None,
            'config': None, 
            'save_train': True,
            'nondetect_val': 99.0, 
            'output_mode': 'default',
            'ref_band': 'mag_i_lsst', 
            'redshift_col': 'redshift', 
            'hdf5_groupname': 'photometry', 
            'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301, 
            'trainfrac': 0.75, 'retrain_full': True,
            'max_basis': 35, 'basis_system': 'cosine', 
            'bumpmin': 0.0, 'bumpmax': 1.0, 'nbump': 50, 
            'sharpmin': 1.0, 'sharpmax': 3.0, 'nsharp': 20, 
            'regression_params': {
                'verbosity': 0, 
                'max_depth': 16, 
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
    
    config_name = os.path.join(file_path, 'INFORM/FZB_CONFIG{}.yaml'.format(index))
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Informer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the train datasets')
    
    PATH = PARSE.parse_args().path
    INDEX = PARSE.parse_args().index
    print('Index: {}'.format(INDEX))
    
    START = time.time()
    main(PATH, INDEX)
    
    END = time.time()
    print('Time: {:.2f} minutes'.format((END - START) / 60))