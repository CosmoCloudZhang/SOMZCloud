import os
import yaml
import time
import argparse

def main(path, index):
    
    # Path
    file_path = os.path.join(path, 'FILE/')
    
    # Config
    config = {
        'FZB_ESTIMATE{}'.format(index): {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'model': 'input_model',
                'output': 'output_data'
            }, 
            'name': None,
            'input': None,
            'output': None,
            'config': None,
            'chunk_size': 250000, 
            'nondetect_val': 99.0, 
            'ref_band': 'mag_i_lsst', 
            'output_mode': 'default',
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
    
    config_name = os.path.join(file_path, 'ESTIMATE/FZB_CONFIG{}.yaml'.format(index))
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Estimator')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--index', type=int, required=True, help='Index of the sample for the modelling')
    
    PATH = PARSE.parse_args().path
    INDEX = PARSE.parse_args().index
    print('Index: {}'.format(INDEX))
    
    START = time.time()
    main(PATH, INDEX)
        
    END = time.time()
    print('Time: {:.2f} minutes'.format((END - START) / 60))