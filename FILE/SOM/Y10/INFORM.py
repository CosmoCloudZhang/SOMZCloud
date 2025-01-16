import os
import h5py
import time
import yaml
import numpy
import argparse


def main(tag, folder):
    '''
    Generate the configuration file for the SOM Informer.
    
    Arguments:
        tag (int) : the tag of the observing conditions
        folder (str) : the base folder containing the datasets
    
    Returns:
        duration (float) : the duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    som_folder = os.path.join(folder, 'SOM/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    os.makedirs(os.path.join(som_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(som_folder, '{}/INFORM/'.format(tag)), exist_ok=True)
    
    # Catalog
    with h5py.File(os.path.join(dataset_folder, '{}/OBSERVATION/OBSERVATION.hdf5'.format(tag)), 'r') as file:
        catalog = {key: file[key][:].astype(numpy.float32) for key in file.keys()}
    
    # Write
    with h5py.File(os.path.join(som_folder, '{}/INFORM/INFORM.hdf5'.format(tag)), 'w') as file:
        file.create_group('photometry')
        file['photometry'].create_dataset('redshift', data=catalog['redshift'])
        
        file['photometry'].create_dataset('mag_u_lsst', data=catalog['mag_u_lsst'])
        file['photometry'].create_dataset('mag_g_lsst', data=catalog['mag_g_lsst'])
        file['photometry'].create_dataset('mag_r_lsst', data=catalog['mag_r_lsst'])
        file['photometry'].create_dataset('mag_i_lsst', data=catalog['mag_i_lsst'])
        file['photometry'].create_dataset('mag_z_lsst', data=catalog['mag_z_lsst'])
        file['photometry'].create_dataset('mag_y_lsst', data=catalog['mag_y_lsst'])
        
        file['photometry'].create_dataset('mag_u_lsst_err', data=catalog['mag_u_lsst_err'])
        file['photometry'].create_dataset('mag_g_lsst_err', data=catalog['mag_g_lsst_err'])
        file['photometry'].create_dataset('mag_r_lsst_err', data=catalog['mag_r_lsst_err'])
        file['photometry'].create_dataset('mag_i_lsst_err', data=catalog['mag_i_lsst_err'])
        file['photometry'].create_dataset('mag_z_lsst_err', data=catalog['mag_z_lsst_err'])
        file['photometry'].create_dataset('mag_y_lsst_err', data=catalog['mag_y_lsst_err'])
    
    # Config
    config = {
        'INFORM': {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'model': 'input_model',
            }, 
            'std_coeff': 0.5, 
            'maptype': 'toroid', 
            'nondetect_val': 99.0, 
            'grid_type': 'hexagonal', 
            'ref_band': 'mag_i_lsst',
            'column_usage': 'colors',
            'som_learning_rate': 0.1,
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
    config_name = os.path.join(som_folder, '{}/INFORM/INFORM.yaml'.format(tag))
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='SOM Informer')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)