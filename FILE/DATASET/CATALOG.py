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
    Generate the photometrically-selected SIMULATION galaxy catalogs.
    
    Arguments:
        folder (str): The base folder of the Augmentation datasets.
    
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
        print(len(observation['redshift_true']))
        with h5py.File(os.path.join(dataset_folder, 'CATALOG/OBSERVATION_{}.hdf5'.format(value)), 'w') as file:
            for key in observation.keys():
                file.create_dataset(key, data=observation[key], dtype=numpy.float32)
    
    # Simulation
    with open(os.path.join(dataset_folder, 'CATALOG/SIMULATE.yaml'), 'r') as file:
        simulation_list = yaml.safe_load(file)['healpix_pixels']
    
    for value in simulation_list:
        print('ID: {}'.format(value))
        catalog = GCRCatalogs.load_catalog('cosmoDC2_v1.1.4_image', config_overwrite={'healpix_pixels': [value]})
        
        simulation = catalog.get_quantities([
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
        print(len(simulation['redshift_true']))
        with h5py.File(os.path.join(dataset_folder, 'CATALOG/SIMULATION_{}.hdf5'.format(value)), 'w') as file:
            for key in simulation.keys():
                file.create_dataset(key, data=simulation[key], dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Catalog')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder containing the datasets')
    
    # Parse
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(FOLDER)