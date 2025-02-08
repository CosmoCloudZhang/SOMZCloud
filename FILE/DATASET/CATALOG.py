import os
import sys
import yaml
sys.path.insert(0, '/global/homes/y/yhzhang/opt/gcr-catalogs/')

import h5py
import time
import numpy
import argparse
import GCRCatalogs

def main(folder):
    '''
    Generate the photometrically-selected galaxy catalogs.
    
    Arguments:
        folder (str): The base folder of the catalogs
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(dataset_folder, 'CATALOG/'), exist_ok=True)
    
    # Observation
    with open(os.path.join(dataset_folder, 'CATALOG/OBSERVE.yaml'), 'r') as file:
        observation_list = yaml.safe_load(file)['healpix_pixels']
    
    for value in observation_list:
        print('ID: {}'.format(value))
        catalog = GCRCatalogs.load_catalog('roman_rubin_2023_v1.1.3_elais', config_overwrite={'healpix_pixels': [value]})
        
        observation = catalog.get_quantities([
            'ra',
            'dec',
            'redshift',
            'mag_u_lsst', 
            'mag_g_lsst',
            'mag_r_lsst',
            'mag_i_lsst',
            'mag_z_lsst',
            'mag_y_lsst', 
            'redshift_true',
            'magnification',
            'size_disk_true', 
            'size_bulge_true', 
            'bulge_to_total_ratio', 
            'ellipticity_disk_true',
            'ellipticity_bulge_true'
        ])
        
        # Save
        with h5py.File(os.path.join(dataset_folder, 'CATALOG/OBSERVATION_{}.hdf5'.format(value)), 'w') as file:
            
            file.create_dataset('mag_u_lsst', data=observation['mag_u_lsst'], dtype=numpy.float32)
            file.create_dataset('mag_g_lsst', data=observation['mag_g_lsst'], dtype=numpy.float32)
            file.create_dataset('mag_r_lsst', data=observation['mag_r_lsst'], dtype=numpy.float32)
            file.create_dataset('mag_i_lsst', data=observation['mag_i_lsst'], dtype=numpy.float32)
            file.create_dataset('mag_z_lsst', data=observation['mag_z_lsst'], dtype=numpy.float32)
            file.create_dataset('mag_y_lsst', data=observation['mag_y_lsst'], dtype=numpy.float32)
            
            file.create_dataset('redshift', data=observation['redshift'], dtype=numpy.float32)
            file.create_dataset('redshift_true', data=observation['redshift_true'], dtype=numpy.float32)
            file.create_dataset('magnification', data=observation['magnification'], dtype=numpy.float32)
            
            file.create_dataset('ra', data=observation['ra'], dtype=numpy.float32)
            file.create_dataset('dec', data=observation['dec'], dtype=numpy.float32)
            file.create_dataset('id', data=numpy.arange(len(observation['redshift'])), dtype=numpy.int32)
            file.create_dataset('value', data=numpy.ones(len(observation['redshift'])) * value, dtype=numpy.int32)
            
            file.create_dataset('major_disk', data=observation['size_disk_true'], dtype=numpy.float32)
            file.create_dataset('major_bulge', data=observation['size_bulge_true'], dtype=numpy.float32)
            file.create_dataset('ellipticity_disk', data=observation['ellipticity_disk_true'], dtype=numpy.float32)
            file.create_dataset('ellipticity_bulge', data=observation['ellipticity_bulge_true'], dtype=numpy.float32)
            file.create_dataset('bulge_to_total_ratio', data=observation['bulge_to_total_ratio'], dtype=numpy.float32)
    
    # Simulation
    with open(os.path.join(dataset_folder, 'CATALOG/SIMULATE.yaml'), 'r') as file:
        simulation_list = yaml.safe_load(file)['healpix_pixels']
    
    for value in simulation_list:
        print('ID: {}'.format(value))
        catalog = GCRCatalogs.load_catalog('cosmoDC2_v1.1.4_image', config_overwrite={'healpix_pixels': [value]})
        
        simulation = catalog.get_quantities([
            'ra',
            'dec',
            'redshift',
            'mag_u_lsst', 
            'mag_g_lsst',
            'mag_r_lsst',
            'mag_i_lsst',
            'mag_z_lsst',
            'mag_y_lsst', 
            'redshift_true',
            'magnification',
            'size_disk_true', 
            'size_bulge_true', 
            'ellipticity_disk_true',
            'ellipticity_bulge_true',
            'bulge_to_total_ratio_i'
        ])
        
        # Save
        with h5py.File(os.path.join(dataset_folder, 'CATALOG/SIMULATION_{}.hdf5'.format(value)), 'w') as file:
            
            file.create_dataset('mag_u_lsst', data=simulation['mag_u_lsst'], dtype=numpy.float32)
            file.create_dataset('mag_g_lsst', data=simulation['mag_g_lsst'], dtype=numpy.float32)
            file.create_dataset('mag_r_lsst', data=simulation['mag_r_lsst'], dtype=numpy.float32)
            file.create_dataset('mag_i_lsst', data=simulation['mag_i_lsst'], dtype=numpy.float32)
            file.create_dataset('mag_z_lsst', data=simulation['mag_z_lsst'], dtype=numpy.float32)
            file.create_dataset('mag_y_lsst', data=simulation['mag_y_lsst'], dtype=numpy.float32)
            
            file.create_dataset('redshift', data=simulation['redshift'], dtype=numpy.float32)
            file.create_dataset('redshift_true', data=simulation['redshift_true'], dtype=numpy.float32)
            file.create_dataset('magnification', data=simulation['magnification'], dtype=numpy.float32)
            
            file.create_dataset('ra', data=simulation['ra'], dtype=numpy.float32)
            file.create_dataset('dec', data=simulation['dec'], dtype=numpy.float32)
            file.create_dataset('id', data=numpy.arange(len(simulation['redshift'])), dtype=numpy.int32)
            file.create_dataset('value', data=numpy.ones(len(simulation['redshift'])) * value, dtype=numpy.int32)
            
            file.create_dataset('major_disk', data=simulation['size_disk_true'], dtype=numpy.float32)
            file.create_dataset('major_bulge', data=simulation['size_bulge_true'], dtype=numpy.float32)
            file.create_dataset('ellipticity_disk', data=simulation['ellipticity_disk_true'], dtype=numpy.float32)
            file.create_dataset('ellipticity_bulge', data=simulation['ellipticity_bulge_true'], dtype=numpy.float32)
            file.create_dataset('bulge_to_total_ratio', data=simulation['bulge_to_total_ratio_i'], dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Catalog')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the catalogs')
    
    # Parse
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(FOLDER)