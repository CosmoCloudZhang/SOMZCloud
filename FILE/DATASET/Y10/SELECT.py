import os
import h5py
import time
import yaml
import numpy
import pandas
import argparse
from photerr import LsstErrorModel


def main(tag, number, folder):
    '''
    Create the selection datasets
    
    Arguments:
        tag (str): The tag of the configuration
        number (int): The number of the selection datasets
        folder (str): The base folder of the selection datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Path
    start = time.time()
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    os.makedirs(os.path.join(dataset_folder, 'CATALOG/'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/SIMULATION/'.format(tag)), exist_ok=True)
    
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/SELECTION/'.format(tag)), exist_ok=True)
    
    # Simulation
    with open(os.path.join(dataset_folder, 'CATALOG/SIMULATE.yaml'), 'r') as file:
        simulation_list = yaml.safe_load(file)['healpix_pixels']
    
    # Load
    simulation_dataset = {
        'mu': numpy.array([]),
        'eta': numpy.array([]),
        'sigma': numpy.array([]),
        'major': numpy.array([]), 
        'minor': numpy.array([]),
        'redshift': numpy.array([]),
        'mag_u_lsst': numpy.array([]),
        'mag_g_lsst': numpy.array([]),
        'mag_r_lsst': numpy.array([]),
        'mag_i_lsst': numpy.array([]),
        'mag_z_lsst': numpy.array([]),
        'mag_y_lsst': numpy.array([]),
        'major_disk': numpy.array([]),
        'major_bulge': numpy.array([]), 
        'magnification': numpy.array([]),
        'ellipticity_disk': numpy.array([]),
        'ellipticity_bulge': numpy.array([]), 
        'bulge_to_total_ratio': numpy.array([])
    }
    
    for value in simulation_list:
        print('ID: {}'.format(value))
        
        with h5py.File(os.path.join(dataset_folder, '{}/SIMULATION/SIMULATION_{}.hdf5'.format(tag, value)), 'r') as file:
            
            simulation_dataset['mu'] = numpy.append(simulation_dataset['mu'], file['mu'][:].astype(numpy.float32), axis=0)
            simulation_dataset['eta'] = numpy.append(simulation_dataset['eta'], file['eta'][:].astype(numpy.float32), axis=0)
            simulation_dataset['sigma'] = numpy.append(simulation_dataset['sigma'], file['sigma'][:].astype(numpy.float32), axis=0)
            
            simulation_dataset['major'] = numpy.append(simulation_dataset['major'], file['major'][:].astype(numpy.float32), axis=0)
            simulation_dataset['minor'] = numpy.append(simulation_dataset['minor'], file['minor'][:].astype(numpy.float32), axis=0)
            
            simulation_dataset['major_disk'] = numpy.append(simulation_dataset['major_disk'], file['major_disk'][:].astype(numpy.float32), axis=0)
            simulation_dataset['major_bulge'] = numpy.append(simulation_dataset['major_bulge'], file['major_bulge'][:].astype(numpy.float32), axis=0)
            
            simulation_dataset['ellipticity_disk'] = numpy.append(simulation_dataset['ellipticity_disk'], file['ellipticity_disk'][:].astype(numpy.float32), axis=0)
            simulation_dataset['ellipticity_bulge'] = numpy.append(simulation_dataset['ellipticity_bulge'], file['ellipticity_bulge'][:].astype(numpy.float32), axis=0)
            
            simulation_dataset['redshift'] = numpy.append(simulation_dataset['redshift'], file['redshift'][:].astype(numpy.float32), axis=0)
            simulation_dataset['magnification'] = numpy.append(simulation_dataset['magnification'], file['magnification'][:].astype(numpy.float32), axis=0)
            simulation_dataset['bulge_to_total_ratio'] = numpy.append(simulation_dataset['bulge_to_total_ratio'], file['bulge_to_total_ratio'][:].astype(numpy.float32), axis=0)
            
            simulation_dataset['mag_u_lsst'] = numpy.append(simulation_dataset['mag_u_lsst'], file['mag_u_lsst'][:].astype(numpy.float32), axis=0)
            simulation_dataset['mag_g_lsst'] = numpy.append(simulation_dataset['mag_g_lsst'], file['mag_g_lsst'][:].astype(numpy.float32), axis=0)
            simulation_dataset['mag_r_lsst'] = numpy.append(simulation_dataset['mag_r_lsst'], file['mag_r_lsst'][:].astype(numpy.float32), axis=0)
            simulation_dataset['mag_i_lsst'] = numpy.append(simulation_dataset['mag_i_lsst'], file['mag_i_lsst'][:].astype(numpy.float32), axis=0)
            simulation_dataset['mag_z_lsst'] = numpy.append(simulation_dataset['mag_z_lsst'], file['mag_z_lsst'][:].astype(numpy.float32), axis=0)
            simulation_dataset['mag_y_lsst'] = numpy.append(simulation_dataset['mag_y_lsst'], file['mag_y_lsst'][:].astype(numpy.float32), axis=0)
    print(len(simulation_dataset['redshift']))
    
    # Redshift
    z1 = 0.05
    z2 = 2.95
    select = (z1 < simulation_dataset['redshift']) & (simulation_dataset['redshift'] < z2)
    
    # Magnitude
    magnitude1 = 15
    magnitude2 = 30
    select = select & (magnitude1 < simulation_dataset['mag_i_lsst']) & (simulation_dataset['mag_i_lsst'] < magnitude2)
    
    for key in simulation_dataset:
        simulation_dataset[key] = simulation_dataset[key][select]
    
    # Error
    error_model = LsstErrorModel(
        nYrObs=10, 
        sigLim=3.0,
        absFlux=True,
        ndMode='sigLim', 
        majorCol='major', 
        minorCol='minor', 
        decorrelate=True,
        extendedType='auto',
        renameDict={
            'u': 'mag_u_lsst',
            'g': 'mag_g_lsst',
            'r': 'mag_r_lsst',
            'i': 'mag_i_lsst',
            'z': 'mag_z_lsst',
            'y': 'mag_y_lsst'
        }
    )
    
    # Selection
    for index in range(1, number + 1):
        print('Index: {:.0f}'.format(index))
        
        with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            size = len(file['photometry']['redshift'][:].astype(numpy.float32))
        
        indices = numpy.random.choice(len(simulation_dataset['redshift']), size=size, replace=False)
        selection_dataset = error_model(pandas.DataFrame(simulation_dataset).iloc[indices])
        
        # Save
        with h5py.File(os.path.join(dataset_folder, '{}/SELECTION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
            file.create_group('photometry')
            file['photometry'].create_dataset('redshift', data=selection_dataset['redshift'][indices], dtype=numpy.float32)
            
            file['photometry'].create_dataset('mag_u_lsst', data=selection_dataset['mag_u_lsst'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_g_lsst', data=selection_dataset['mag_g_lsst'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_r_lsst', data=selection_dataset['mag_r_lsst'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_i_lsst', data=selection_dataset['mag_i_lsst'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_z_lsst', data=selection_dataset['mag_z_lsst'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_y_lsst', data=selection_dataset['mag_y_lsst'][indices], dtype=numpy.float32)
            
            file['photometry'].create_dataset('mag_u_lsst_err', data=selection_dataset['mag_u_lsst_err'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_g_lsst_err', data=selection_dataset['mag_g_lsst_err'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_r_lsst_err', data=selection_dataset['mag_r_lsst_err'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_i_lsst_err', data=selection_dataset['mag_i_lsst_err'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_z_lsst_err', data=selection_dataset['mag_z_lsst_err'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_y_lsst_err', data=selection_dataset['mag_y_lsst_err'][indices], dtype=numpy.float32)
            
            file.create_group('morphology')
            file['morphology'].create_dataset('mu', data=selection_dataset['mu'][indices], dtype=numpy.float32)
            file['morphology'].create_dataset('eta', data=selection_dataset['eta'][indices], dtype=numpy.float32)
            file['morphology'].create_dataset('sigma', data=selection_dataset['sigma'][indices], dtype=numpy.float32)
            
            file['morphology'].create_dataset('major', data=selection_dataset['major'][indices], dtype=numpy.float32)
            file['morphology'].create_dataset('minor', data=selection_dataset['minor'][indices], dtype=numpy.float32)
            
            file['morphology'].create_dataset('major_disk', data=selection_dataset['major_disk'][indices], dtype=numpy.float32)
            file['morphology'].create_dataset('major_bulge', data=selection_dataset['major_bulge'][indices], dtype=numpy.float32)
            
            file['morphology'].create_dataset('ellipticity_disk', data=selection_dataset['ellipticity_disk'][indices], dtype=numpy.float32)
            file['morphology'].create_dataset('ellipticity_bulge', data=selection_dataset['ellipticity_bulge'][indices], dtype=numpy.float32)
            file['morphology'].create_dataset('bulge_to_total_ratio', data=selection_dataset['bulge_to_total_ratio'][indices], dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Selection Datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the selection datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the selection datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)