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
        tag (str): The tag of observing conditions
        number (int): The number of the selection datasets
        folder (str): The base folder containing the datasets
    
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
    catalog = {
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
            
            catalog['mu'] = numpy.append(catalog['mu'], file['mu'][:].astype(numpy.float32), axis=0)
            catalog['eta'] = numpy.append(catalog['eta'], file['eta'][:].astype(numpy.float32), axis=0)
            catalog['sigma'] = numpy.append(catalog['sigma'], file['sigma'][:].astype(numpy.float32), axis=0)
            
            catalog['major'] = numpy.append(catalog['major'], file['major'][:].astype(numpy.float32), axis=0)
            catalog['minor'] = numpy.append(catalog['minor'], file['minor'][:].astype(numpy.float32), axis=0)
            
            catalog['major_disk'] = numpy.append(catalog['major_disk'], file['major_disk'][:].astype(numpy.float32), axis=0)
            catalog['major_bulge'] = numpy.append(catalog['major_bulge'], file['major_bulge'][:].astype(numpy.float32), axis=0)
            
            catalog['ellipticity_disk'] = numpy.append(catalog['ellipticity_disk'], file['ellipticity_disk'][:].astype(numpy.float32), axis=0)
            catalog['ellipticity_bulge'] = numpy.append(catalog['ellipticity_bulge'], file['ellipticity_bulge'][:].astype(numpy.float32), axis=0)
            
            catalog['redshift'] = numpy.append(catalog['redshift'], file['redshift'][:].astype(numpy.float32), axis=0)
            catalog['magnification'] = numpy.append(catalog['magnification'], file['magnification'][:].astype(numpy.float32), axis=0)
            catalog['bulge_to_total_ratio'] = numpy.append(catalog['bulge_to_total_ratio'], file['bulge_to_total_ratio'][:].astype(numpy.float32), axis=0)
            
            catalog['mag_u_lsst'] = numpy.append(catalog['mag_u_lsst'], file['mag_u_lsst'][:].astype(numpy.float32), axis=0)
            catalog['mag_g_lsst'] = numpy.append(catalog['mag_g_lsst'], file['mag_g_lsst'][:].astype(numpy.float32), axis=0)
            catalog['mag_r_lsst'] = numpy.append(catalog['mag_r_lsst'], file['mag_r_lsst'][:].astype(numpy.float32), axis=0)
            catalog['mag_i_lsst'] = numpy.append(catalog['mag_i_lsst'], file['mag_i_lsst'][:].astype(numpy.float32), axis=0)
            catalog['mag_z_lsst'] = numpy.append(catalog['mag_z_lsst'], file['mag_z_lsst'][:].astype(numpy.float32), axis=0)
            catalog['mag_y_lsst'] = numpy.append(catalog['mag_y_lsst'], file['mag_y_lsst'][:].astype(numpy.float32), axis=0)
    print(len(catalog['redshift']))
    
    # Redshift
    z1 = 0.05
    z2 = 2.95
    select = (z1 < catalog['redshift']) & (catalog['redshift'] < z2)
    
    # Magnitude
    magnitude1 = 15
    magnitude2 = 30
    select = select & (magnitude1 < catalog['mag_i_lsst']) & (catalog['mag_i_lsst'] < magnitude2)
    
    for key in catalog:
        catalog[key] = catalog[key][select]
    
    # Error
    error_model = LsstErrorModel(
        nYrObs=1, 
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
            length = len(file['photometry']['redshift'][:].astype(numpy.float32))
        
        indices = numpy.random.choice(len(catalog['redshift']), length, replace=False)
        table = error_model(pandas.DataFrame(catalog).iloc[indices])
        
        # Save
        with h5py.File(os.path.join(dataset_folder, '{}/SELECTION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
            file.create_group('photometry')
            file['photometry'].create_dataset('redshift', data=table['redshift'][indices], dtype=numpy.float32)
            
            file['photometry'].create_dataset('mag_u_lsst', data=table['mag_u_lsst'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_g_lsst', data=table['mag_g_lsst'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_r_lsst', data=table['mag_r_lsst'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_i_lsst', data=table['mag_i_lsst'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_z_lsst', data=table['mag_z_lsst'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_y_lsst', data=table['mag_y_lsst'][indices], dtype=numpy.float32)
            
            file['photometry'].create_dataset('mag_u_lsst_err', data=table['mag_u_lsst_err'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_g_lsst_err', data=table['mag_g_lsst_err'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_r_lsst_err', data=table['mag_r_lsst_err'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_i_lsst_err', data=table['mag_i_lsst_err'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_z_lsst_err', data=table['mag_z_lsst_err'][indices], dtype=numpy.float32)
            file['photometry'].create_dataset('mag_y_lsst_err', data=table['mag_y_lsst_err'][indices], dtype=numpy.float32)
            
            file.create_group('morphology')
            file['morphology'].create_dataset('mu', data=table['mu'][indices], dtype=numpy.float32)
            file['morphology'].create_dataset('eta', data=table['eta'][indices], dtype=numpy.float32)
            file['morphology'].create_dataset('sigma', data=table['sigma'][indices], dtype=numpy.float32)
            
            file['morphology'].create_dataset('major', data=table['major'][indices], dtype=numpy.float32)
            file['morphology'].create_dataset('minor', data=table['minor'][indices], dtype=numpy.float32)
            
            file['morphology'].create_dataset('major_disk', data=table['major_disk'][indices], dtype=numpy.float32)
            file['morphology'].create_dataset('major_bulge', data=table['major_bulge'][indices], dtype=numpy.float32)
            
            file['morphology'].create_dataset('ellipticity_disk', data=table['ellipticity_disk'][indices], dtype=numpy.float32)
            file['morphology'].create_dataset('ellipticity_bulge', data=table['ellipticity_bulge'][indices], dtype=numpy.float32)
            file['morphology'].create_dataset('bulge_to_total_ratio', data=table['bulge_to_total_ratio'][indices], dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Selection Datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of observing conditions')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the selection datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder containing the datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)