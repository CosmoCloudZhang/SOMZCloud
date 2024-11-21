import os
import h5py
import time
import numpy
import pandas
import argparse
from photerr import LsstErrorModel

def catalog(folder, directory):
    start = time.time()
    data_folder = os.path.join(folder, 'DATA')
    name_list = numpy.sort(os.listdir(directory))
    
    data = {
        'major': numpy.array([]),
        'minor': numpy.array([]),
        'redshift': numpy.array([]),
        'mag_u_lsst': numpy.array([]),
        'mag_g_lsst': numpy.array([]),
        'mag_r_lsst': numpy.array([]),
        'mag_i_lsst': numpy.array([]),
        'mag_z_lsst': numpy.array([]),
        'mag_y_lsst': numpy.array([])
    }
    
    for name in name_list:
        print('Name: {}'.format(name))
        
        with h5py.File(os.path.join(directory, name), 'r') as file:
            key_list = numpy.sort([key for key in file.keys() if key != 'metaData'])
            for key in key_list:
                if list(file[key].keys()):
                    
                    bulge_fraction = file[key]['bulge_frac'][:].astype(numpy.float32)
                    disk_major = file[key]['diskHalfLightRadiusArcsec'][:].astype(numpy.float32)
                    bulge_major = file[key]['spheroidHalfLightRadiusArcsec'][:].astype(numpy.float32)
                    
                    disk_minor = disk_major * file[key]['diskAxisRatio'][:].astype(numpy.float32)
                    bulge_minor = bulge_major * file[key]['spheroidAxisRatio'][:].astype(numpy.float32)
                    
                    data['major'] = bulge_fraction * bulge_major + (1 - bulge_fraction) * disk_major
                    data['minor'] = bulge_fraction * bulge_minor + (1 - bulge_fraction) * disk_minor
                    
                    data['redshift'] = numpy.concatenate((data['redshift'], file[key]['redshift'][:].astype(numpy.float32)))
                    
                    data['mag_u_lsst'] = numpy.concatenate((data['mag_u_lsst'], file[key]['LSST_obs_u'][:].astype(numpy.float32)))
                    
                    data['mag_g_lsst'] = numpy.concatenate((data['mag_g_lsst'], file[key]['LSST_obs_g'][:].astype(numpy.float32)))
                    
                    data['mag_r_lsst'] = numpy.concatenate((data['mag_r_lsst'], file[key]['LSST_obs_r'][:].astype(numpy.float32)))
                    
                    data['mag_i_lsst'] = numpy.concatenate((data['mag_i_lsst'], file[key]['LSST_obs_i'][:].astype(numpy.float32)))
                    
                    data['mag_z_lsst'] = numpy.concatenate((data['mag_z_lsst'], file[key]['LSST_obs_z'][:].astype(numpy.float32)))
                    
                    data['mag_y_lsst'] = numpy.concatenate((data['mag_y_lsst'], file[key]['LSST_obs_y'][:].astype(numpy.float32)))
    # Save
    os.makedirs(os.path.join(data_folder, 'AUGMENTATION'), exist_ok=True)
    with h5py.File(os.path.join(data_folder, 'AUGMENTATION/CATALOG.hdf5'), 'w') as file:
        for key in data.keys():
            file.create_dataset(key, data=data[key])
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


def data(folder):
    start = time.time()
    data_folder = os.path.join(folder, 'DATA')
    
    # Load
    catalog = pandas.read_hdf(os.path.join(data_folder, 'AUGMENTATION/CATALOG.hdf5'))
    print(catalog.columns)
    # Error
    error_model = LsstErrorModel(
        nYrObs=1, 
        majorCol='major', 
        minorCol='minor', 
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
    
    table = error_model(catalog, random_state=42)
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Augmentation datasets')
    PARSE.add_argument('--count', type=int, required=True, help='The number of processes')
    PARSE.add_argument('--number', type=int, required=True, help='The number of Augmentation datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the Augmentation datasets')
    PARSE.add_argument('--directory', type=str, required=True, help='The directory of the Roman-Rubin simulation catalogs')
    
    COUNT = PARSE.parse_args().count
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    DIRECTORY = PARSE.parse_args().directory
    
    # Output
    CATALOG = catalog(FOLDER, DIRECTORY)