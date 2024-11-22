import os
import h5py
import time
import numpy
import pandas
import argparse
from photerr import LsstErrorModel

def catalog(folder, directory):
    
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
    for name in name_list[:1]:
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
                    
                    data['major'] = numpy.concatenate((data['major'], bulge_fraction * bulge_major + (1 - bulge_fraction) * disk_major))
                    data['minor'] = numpy.concatenate((data['minor'], bulge_fraction * bulge_minor + (1 - bulge_fraction) * disk_minor))
                    
                    data['redshift'] = numpy.concatenate((data['redshift'], file[key]['redshift'][:].astype(numpy.float32)))
                    
                    data['mag_u_lsst'] = numpy.concatenate((data['mag_u_lsst'], file[key]['LSST_obs_u'][:].astype(numpy.float32)))
                    
                    data['mag_g_lsst'] = numpy.concatenate((data['mag_g_lsst'], file[key]['LSST_obs_g'][:].astype(numpy.float32)))
                    
                    data['mag_r_lsst'] = numpy.concatenate((data['mag_r_lsst'], file[key]['LSST_obs_r'][:].astype(numpy.float32)))
                    
                    data['mag_i_lsst'] = numpy.concatenate((data['mag_i_lsst'], file[key]['LSST_obs_i'][:].astype(numpy.float32)))
                    
                    data['mag_z_lsst'] = numpy.concatenate((data['mag_z_lsst'], file[key]['LSST_obs_z'][:].astype(numpy.float32)))
                    
                    data['mag_y_lsst'] = numpy.concatenate((data['mag_y_lsst'], file[key]['LSST_obs_y'][:].astype(numpy.float32)))
    
    # Save
    data_folder = os.path.join(folder, 'DATASET')
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
    # Start
    start = time.time()
    
    # Load
    data_folder = os.path.join(folder, 'DATASET')
    with h5py.File(os.path.join(data_folder, 'AUGMENTATION/CATALOG.hdf5'), 'r') as file:
        catalog = pandas.DataFrame({key: file[key][:].astype(numpy.float32) for key in file.keys()})
    
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
    
    # Random
    seed = 100
    numpy.random.seed(seed)
    table = error_model(catalog, random_state=seed)
    
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
        catalog['snr_{}'.format(band)] = 2.5 / numpy.log(10) / mag_err + epsilon
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
    data = {'photometry': {}}
    data['photometry']['redshift'] = catalog['redshift'][select]
    
    for band in band_list:
        
        # Photometry
        mag = catalog['mag_{}'.format(band)].values
        mag_err = catalog['mag_err_{}'.format(band)].values
        
        # Mask
        mask = catalog['snr_{}'.format(band)] < 3.0
        mag_err[mask] = 99.0
        mag[mask] = 99.0
        
        # Save
        data['photometry']['mag_{}'.format(band)] = mag[select]
        data['photometry']['mag_err_{}'.format(band)] = mag_err[select]
    
    # Save
    os.makedirs(os.path.join(data_folder, 'AUGMENTATION'), exist_ok=True)
    with h5py.File(os.path.join(data_folder, 'AUGMENTATION/DATA.hdf5'), 'w') as file:
        for key, value in data['photometry'].items():
            file.create_dataset(key, data=value)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


def augmentation(count, number, folder):
    
    # Data store
    start = time.time()
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Path
    plot_path = os.path.join(path, 'PLOT/')
    data_path = os.path.join(path, 'DATA/')
    
    # Augment datasets
    augment_name = os.path.join(data_path, 'SAMPLE/AUGMENT_SAMPLE.hdf5')
    augment_data = data_store.read_file(key='augment_data', path=augment_name, handle_class=core.data.TableHandle)()
    
    z_augment = augment_data['photometry']['redshift']
    mag_augment = augment_data['photometry']['mag_i_lsst']
    color_augment = augment_data['photometry']['mag_g_lsst'] - augment_data['photometry']['mag_z_lsst']
    
    # Input datasets
    input_name = os.path.join(data_path, 'SAMPLE/INPUT_SAMPLE{}.hdf5'.format(index))
    input_data = data_store.read_file(key='input_data', path=input_name, handle_class=core.data.TableHandle)()
    
    z_input = input_data['photometry']['redshift']
    mag_input = input_data['photometry']['mag_i_lsst']
    color_input = input_data['photometry']['mag_g_lsst'] - input_data['photometry']['mag_z_lsst']
    
    # Test datasets
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)()
    
    z_test = test_data['photometry']['redshift']
    mag_test = test_data['photometry']['mag_i_lsst']
    color_test = test_data['photometry']['mag_g_lsst'] - test_data['photometry']['mag_z_lsst']
    
    # Bin Datasets
    z1 = 0.0
    z2 = 3.0
    z_bin_size = 50
    z_bin = numpy.linspace(z1, z2, z_bin_size + 1)
    
    mag1 = 14.0
    mag2 = 26.0
    mag_bin_size = 50
    mag_bin = numpy.linspace(mag1, mag2, mag_bin_size + 1)
    
    color1 = -1.0
    color2 = +6.0
    color_bin_size = 50
    color_bin = numpy.linspace(color1, color2, color_bin_size + 1)
    
    bin_datasets = [z_bin, mag_bin, color_bin]
    test_datasets = [z_test, mag_test, color_test]
    input_datasets = [z_input, mag_input, color_input]
    augment_datasets = [z_augment, mag_augment, color_augment]
    
    z_bin, mag_bin, color_bin = bin_datasets
    z_input, mag_input, color_input = input_datasets
    z_augment, mag_augment, color_augment = augment_datasets
    
    z1 = z_bin.min()
    z2 = z_bin.max()
    z_delta = (z2 - z1) / len(z_bin)
    z_data = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, len(z_bin) - 1)
    
    mag1 = mag_bin.min()
    mag2 = mag_bin.max()
    mag_delta = (mag2 - mag1) / len(mag_bin)
    mag_data = numpy.linspace(mag1 + mag_delta / 2, mag2 - mag_delta / 2, len(mag_bin) - 1)
    
    color1 = color_bin.min()
    color2 = color_bin.max()
    color_delta = (color2 - color1) / len(color_bin)
    color_data = numpy.linspace(color1 + color_delta / 2, color2 - color_delta / 2, len(color_bin) - 1)
    
    pdf, edges = numpy.histogramdd([z_input, mag_input, color_input], bins=[z_bin, mag_bin, color_bin], density=True)
    
    sigma = numpy.quantile(pdf[pdf > 0], 0.01)
    factor = numpy.log(1 + numpy.exp(- numpy.square(pdf / sigma)))
    
    weight = scipy.interpolate.interpn(points=(z_data, mag_data, color_data), values=factor, xi=(z_augment, mag_augment, color_augment), method='linear', bounds_error=False, fill_value=0.0)
    weight = weight / numpy.sum(weight)
    
    count = len(z_input) // 4
    index = numpy.arange(len(z_augment))
    index_sample = numpy.random.choice(index, size=count, replace=True, p=weight)
    
    select_sample = numpy.zeros_like(z_augment, dtype=bool)
    select_sample[index_sample] = True
    
    return 0


def main(count, number, folder, directory):
    start = time.time()
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    catalog(folder, directory)
    
    data(folder)
    
    augmentation(count, number, folder)
    
    # Return
    print('Total Time: {:.2f} minutes'.format(duration))
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
    OUTPUT = main(COUNT, NUMBER, FOLDER, DIRECTORY)