import os
import h5py
import time
import yaml
import numpy
import pandas
import argparse
from photerr import LsstErrorModel


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
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/SOM/'.format(tag)), exist_ok=True)
    
    # Error
    error_model = LsstErrorModel(
        sigLim=1.0,
        absFlux=True,
        ndMode='sigLim', 
        majorCol='major', 
        minorCol='minor', 
        decorrelate=True,
        extendedType='auto',
        nYrObs=int(tag[1:]), 
        renameDict={
            'u': 'mag_u_lsst',
            'g': 'mag_g_lsst',
            'r': 'mag_r_lsst',
            'i': 'mag_i_lsst',
            'z': 'mag_z_lsst',
            'y': 'mag_y_lsst'
        }
    )
    
    # Observation
    with h5py.File(os.path.join(dataset_folder, '{}/OBSERVATION/OBSERVATION.hdf5'.format(tag)), 'r') as file:
        observation_dataset = {key: file[key][...] for key in file.keys()}
    
    # Inform
    inform_dataset = dict(error_model(pandas.DataFrame(observation_dataset), random_state=0))
    
    # Filter
    snr = 15    
    magnitude1 = 16
    magnitude2 = error_model.getLimitingMags(nSigma=snr, coadded=True, aperture=0)
    filter = (magnitude1 < inform_dataset['mag_i_lsst']) & (inform_dataset['mag_i_lsst'] < magnitude2)
    
    for key in inform_dataset.keys():
        inform_dataset[key] = inform_dataset[key][filter]
    
    # Write
    with h5py.File(os.path.join(dataset_folder, '{}/SOM/INFORM.hdf5'.format(tag)), 'w') as file:
        file.create_group('photometry')
        file['photometry'].create_dataset('redshift', data=inform_dataset['redshift'], dtype=numpy.float32)
        file['photometry'].create_dataset('redshift_true', data=inform_dataset['redshift_true'], dtype=numpy.float32)
        
        file['photometry'].create_dataset('mag_u_lsst', data=inform_dataset['mag_u_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_g_lsst', data=inform_dataset['mag_g_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_r_lsst', data=inform_dataset['mag_r_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_i_lsst', data=inform_dataset['mag_i_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_z_lsst', data=inform_dataset['mag_z_lsst'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_y_lsst', data=inform_dataset['mag_y_lsst'], dtype=numpy.float32)
        
        file['photometry'].create_dataset('mag_u_lsst_err', data=inform_dataset['mag_u_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_g_lsst_err', data=inform_dataset['mag_g_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_r_lsst_err', data=inform_dataset['mag_r_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_i_lsst_err', data=inform_dataset['mag_i_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_z_lsst_err', data=inform_dataset['mag_z_lsst_err'], dtype=numpy.float32)
        file['photometry'].create_dataset('mag_y_lsst_err', data=inform_dataset['mag_y_lsst_err'], dtype=numpy.float32)
        
        file.create_group('morphology')
        file['morphology'].create_dataset('id', data=inform_dataset['id'], dtype=numpy.int32)
        file['morphology'].create_dataset('value', data=inform_dataset['value'], dtype=numpy.int32)
        
        file['morphology'].create_dataset('ra', data=inform_dataset['ra'], dtype=numpy.float32)
        file['morphology'].create_dataset('dec', data=inform_dataset['dec'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('mu', data=inform_dataset['mu'], dtype=numpy.float32)
        file['morphology'].create_dataset('eta', data=inform_dataset['eta'], dtype=numpy.float32)
        file['morphology'].create_dataset('sigma', data=inform_dataset['sigma'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('major', data=inform_dataset['major'], dtype=numpy.float32)
        file['morphology'].create_dataset('minor', data=inform_dataset['minor'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('major_disk', data=inform_dataset['major_disk'], dtype=numpy.float32)
        file['morphology'].create_dataset('major_bulge', data=inform_dataset['major_bulge'], dtype=numpy.float32)
        
        file['morphology'].create_dataset('ellipticity_disk', data=inform_dataset['ellipticity_disk'], dtype=numpy.float32)
        file['morphology'].create_dataset('ellipticity_bulge', data=inform_dataset['ellipticity_bulge'], dtype=numpy.float32)
        file['morphology'].create_dataset('bulge_to_total_ratio', data=inform_dataset['bulge_to_total_ratio'], dtype=numpy.float32)
    
    # Config
    config = {
        'INFORM': {
            'aliases': {
                'name': 'input_name',
                'input': 'input_data', 
                'model': 'input_model',
            }, 
            'n_rows': 180,
            'n_columns': 180, 
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
    config_name = os.path.join(dataset_folder, '{}/SOM/INFORM.yaml'.format(tag))
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