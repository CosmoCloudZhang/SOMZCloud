import os
import h5py
import time
import numpy
import scipy
import pandas
import argparse
from photerr import LsstErrorModel

def catalog(directory):
    '''
    Load the Roman-Rubin simulation catalogs
    
    Arguments:
        directory (str): The directory of the Roman-Rubin simulation catalogs
    
    Returns:
        data (dict): The Roman-Rubin simulation catalogs
    '''
    # Start
    start = time.time()
    
    # Data
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
    
    # Load
    name_list = numpy.sort(os.listdir(directory))
    for name in name_list:
        print('Name: {}'.format(name))
        
        with h5py.File(os.path.join(directory, name), 'r') as file:
            key_list = numpy.sort([key for key in file.keys() if key != 'metaData'])
            for key in key_list:
                if list(file[key].keys()):
                    
                    bulge_fraction = file[key]['bulge_frac'][:].astype(numpy.float32)
                    disk_radius = file[key]['diskHalfLightRadiusArcsec'][:].astype(numpy.float32)
                    bulge_radius = file[key]['spheroidHalfLightRadiusArcsec'][:].astype(numpy.float32)
                    
                    ellipticity = file[key]['totalEllipticity'][:].astype(numpy.float32)
                    radius = bulge_fraction * bulge_radius + (1 - bulge_fraction) * disk_radius
                    
                    data['major'] = numpy.append(data['major'], radius / numpy.sqrt((1 - ellipticity) / (1 + ellipticity)))
                    data['minor'] = numpy.append(data['minor'], radius * numpy.sqrt((1 - ellipticity) / (1 + ellipticity)))
                    
                    data['redshift'] = numpy.append(data['redshift'], file[key]['redshift'][:].astype(numpy.float32))
                    
                    data['mag_u_lsst'] = numpy.append(data['mag_u_lsst'], file[key]['LSST_obs_u'][:].astype(numpy.float32))
                    
                    data['mag_g_lsst'] = numpy.append(data['mag_g_lsst'], file[key]['LSST_obs_g'][:].astype(numpy.float32))
                    
                    data['mag_r_lsst'] = numpy.append(data['mag_r_lsst'], file[key]['LSST_obs_r'][:].astype(numpy.float32))
                    
                    data['mag_i_lsst'] = numpy.append(data['mag_i_lsst'], file[key]['LSST_obs_i'][:].astype(numpy.float32))
                    
                    data['mag_z_lsst'] = numpy.append(data['mag_z_lsst'], file[key]['LSST_obs_z'][:].astype(numpy.float32))
                    
                    data['mag_y_lsst'] = numpy.append(data['mag_y_lsst'], file[key]['LSST_obs_y'][:].astype(numpy.float32))
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return data


def dataset(folder):
    '''
    Create the Roman-Rubin simulation datasets
    
    Arguments:
        folder (str): The base folder of the Augmentation datasets
    
    Returns:
        data (dict): The Roman-Rubin simulation datasets
    '''
    # Start
    data_folder = os.path.join(folder, 'DATASET')
    
    # Load
    with h5py.File(os.path.join(data_folder, 'AUGMENTATION/CATALOG.hdf5'), 'r') as file:
        catalog = pandas.DataFrame({key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()})
    
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
    print(error_model.getLimitingMags())
    table = error_model(catalog)
    
    # Band
    band_list = ['u_lsst', 'g_lsst', 'r_lsst', 'i_lsst', 'z_lsst', 'y_lsst']
    for band in band_list:
        
        # Photometry
        mag = catalog['mag_{}'.format(band)].values
        mag_err = table['mag_{}_err'.format(band)].values
        
        # Sampling
        epsilon = numpy.random.normal(0, 1, mag_err.shape)
        mag = mag - 2.5 * numpy.log10(numpy.abs(1 + epsilon * mag_err * numpy.log(10) / 2.5))
        
        catalog['mag_{}'.format(band)] = mag
        catalog['mag_err_{}'.format(band)] = mag_err
        catalog['snr_{}'.format(band)] = 2.5 / numpy.log(10) / mag_err
    # Redshift
    z1 = 0.0
    z2 = 3.0
    select = (z1 < catalog['redshift']) & (catalog['redshift'] < z2)
    
    # Magnitude
    mag1 = 16.0
    mag2 = 26.0
    select = select & (mag1 < catalog['mag_i_lsst']) & (catalog['mag_i_lsst'] < mag2)
    
    # SNR
    snr1 = 3.0
    snr2 = 5.0
    select = select & (snr1 < catalog['snr_r_lsst']) & (catalog['snr_i_lsst'] > snr2)
    
    # Data
    data = {}
    data['redshift'] = catalog['redshift'].values[select]
    
    for band in band_list:
        
        # Photometry
        mag = catalog['mag_{}'.format(band)].values
        mag_err = catalog['mag_err_{}'.format(band)].values
        
        # Mask
        mask = catalog['snr_{}'.format(band)] < 3.0
        mag_err[mask] = 30.0
        mag[mask] = 30.0
        
        # Save
        data['mag_{}'.format(band)] = mag[select]
        data['mag_err_{}'.format(band)] = mag_err[select]
    return data


def augmentation(index, folder):
    '''
    Create the Roman-Rubin simulation augmentation datasets
    
    Arguments:
        index (int): The index of the Augmentation datasets
        folder (str): The base folder of the Augmentation datasets
    
    Returns:
        data (dict): The Roman-Rubin simulation augmentation datasets
    '''
    # Load
    data_folder = os.path.join(folder, 'DATASET')
    with h5py.File(os.path.join(data_folder, 'AUGMENTATION/DATA.hdf5'), 'r') as file:
        data = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
    z_data = data['redshift']
    mag_data = data['mag_i_lsst']
    color_data = data['mag_g_lsst'] - data['mag_z_lsst']
    
    # Selection
    with h5py.File(os.path.join(data_folder, 'SELECTION/DATA{}.hdf5'.format(index + 1)), 'r') as file:
        selection_data = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
    z_selection = selection_data['redshift']
    mag_selection = selection_data['mag_i_lsst']
    color_selection = selection_data['mag_g_lsst'] - selection_data['mag_z_lsst']
    
    # Bin
    z1 = 0.0
    z2 = 3.0
    z_size = 50
    z_delta = (z2 - z1) / z_size
    z_bin = numpy.linspace(z1, z2, z_size + 1)
    z_grid = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, z_size)
    
    mag1 = 16.0
    mag2 = 26.0
    mag_size = 50
    mag_delta = (mag2 - mag1) / mag_size
    mag_bin = numpy.linspace(mag1, mag2, mag_size + 1)
    mag_grid = numpy.linspace(mag1 + mag_delta / 2, mag2 - mag_delta / 2, mag_size)
    
    color1 = -2.0
    color2 = +8.0
    color_size = 50
    color_delta = (color2 - color1) / color_size
    color_bin = numpy.linspace(color1, color2, color_size + 1)
    color_grid = numpy.linspace(color1 + color_delta / 2, color2 - color_delta / 2, color_size)
    
    # PDF
    pdf = numpy.histogramdd([z_selection, mag_selection, color_selection], bins=[z_bin, mag_bin, color_bin], density=True)[0]
    factor = numpy.log(1 + numpy.exp(- numpy.square(pdf / numpy.quantile(pdf[pdf > 0], 0.01))))
    
    weight = scipy.interpolate.interpn(points=(z_grid, mag_grid, color_grid), values=factor, xi=numpy.stack([z_data, mag_data, color_data], axis=1), method='slinear', bounds_error=False, fill_value=0.0)
    weight = numpy.abs(weight) / numpy.sum(weight)
    
    fraction1 = 0.10
    fraction2 = 0.40
    fraction = numpy.random.uniform(fraction1, fraction2)
    
    count = numpy.round(len(z_selection) * fraction, decimals=0).astype(numpy.int32)
    indices = numpy.random.choice(numpy.arange(len(z_data)), size=count, replace=True, p=weight)
    
    return {key: value[indices] for key, value in data.items()}


def main(number, folder, directory):
    start = time.time()
    data_folder = os.path.join(folder, 'DATASET')
    
    # Path
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(os.path.join(data_folder, 'AUGMENTATION'), exist_ok=True)
    
    # Catalog
    data = catalog(directory)
    
    with h5py.File(os.path.join(data_folder, 'AUGMENTATION/CATALOG.hdf5'), 'w') as file:
        file.create_group('photometry')
        
        for key in data.keys():
            file['photometry'].create_dataset(key, data=data[key])
    
    # Dataset
    data = dataset(folder)
    
    with h5py.File(os.path.join(data_folder, 'AUGMENTATION/DATA.hdf5'), 'w') as file:
        file.create_group('photometry')
        
        for key, value in data.items():
            file['photometry'].create_dataset(key, data=value)
    
    # Augmentation
    for index in range(number):
        print('Index: {}'.format(index + 1))
        
        data = augmentation(index, folder)
        
        with h5py.File(os.path.join(data_folder, 'AUGMENTATION/DATA{}.hdf5'.format(index + 1)), 'w') as file:
            file.create_group('photometry')
            
            for key, value in data.items():
                file['photometry'].create_dataset(key, data=value)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Total Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Augmentation datasets')
    PARSE.add_argument('--number', type=int, required=True, help='The number of Augmentation datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the Augmentation datasets')
    PARSE.add_argument('--directory', type=str, required=True, help='The directory of the Roman-Rubin simulation catalogs')
    
    # Parse
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    DIRECTORY = PARSE.parse_args().directory
    
    # Output
    OUTPUT = main(NUMBER, FOLDER, DIRECTORY)