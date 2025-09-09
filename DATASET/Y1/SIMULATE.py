import os
import h5py
import time
import yaml
import numpy
import pandas
import argparse
from photerr import LsstErrorModel


def main(tag, index, folder):
    '''
    Generate the simulation datasets
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of the simulation datasets
        folder (str): The base folder of the simulation datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    random_generator = numpy.random.default_rng(index)
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(dataset_folder, 'CATALOG/'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/SIMULATION/'.format(tag)), exist_ok=True)
    
    # Observation
    with open(os.path.join(dataset_folder, 'CATALOG/OBSERVE.yaml'), 'r') as file:
        observation_list = yaml.safe_load(file)['healpix_pixels']
    
    # Simulation
    with open(os.path.join(dataset_folder, 'CATALOG/SIMULATE.yaml'), 'r') as file:
        simulation_list = yaml.safe_load(file)['healpix_pixels']
        simulation_list = numpy.sort(random_generator.choice(simulation_list, size=len(observation_list), replace=False))
    
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
    
    # Variables
    fwhm_list = [0.92, 0.87, 0.83, 0.80, 0.78, 0.76]
    band_list = ['u_lsst', 'g_lsst', 'r_lsst', 'i_lsst', 'z_lsst', 'y_lsst']
    psf_fwhm = {'psf_{}'.format(band): fwhm for band, fwhm in zip(band_list, fwhm_list)}
    exposure = {key: value * error_model.params.nYrObs for key, value in error_model.params.nVisYr.items()}
    
    # Loop
    simulation = {}
    
    for value in simulation_list:
        print('ID: {}'.format(value))
        
        # Load
        with h5py.File(os.path.join(dataset_folder, 'CATALOG/SIMULATION_{}.hdf5'.format(value)), 'r') as file:
            catalog = {key: file[key][...] for key in file.keys()}
        
        major_disk = catalog['major_disk'] * numpy.sqrt(catalog['magnification'])
        major_bulge = catalog['major_bulge'] * numpy.sqrt(catalog['magnification'])
        
        minor_disk = major_disk * (1 - catalog['ellipticity_disk']) / (1 + catalog['ellipticity_disk'])
        minor_bulge = major_bulge * (1 - catalog['ellipticity_bulge']) / (1 + catalog['ellipticity_bulge'])
        
        fraction = catalog['bulge_to_total_ratio']
        catalog['major'] = fraction * major_bulge + (1 - fraction) * major_disk
        catalog['minor'] = fraction * minor_bulge + (1 - fraction) * minor_disk
        
        factor_disk = 1.46
        factor_bulge = 4.66
        
        radius_disk = factor_disk * numpy.sqrt(major_disk * minor_disk)
        radius_bulge = factor_bulge * numpy.sqrt(major_bulge * minor_bulge)
        catalog['radius'] = numpy.sqrt(fraction * numpy.square(radius_bulge) + (1 - fraction) * numpy.square(radius_disk))
        
        catalog = error_model(pandas.DataFrame(catalog), random_state=index)
        
        # Mu
        flux0 = 3631e6
        flux1 = flux0 * numpy.power(10, -0.4 * catalog['mag_r_lsst'])
        flux2 = flux0 * numpy.power(10, -0.4 * catalog['mag_i_lsst'])
        
        error1 = 2.5 / numpy.log(10) * catalog['mag_r_lsst_err'].values * flux1 * numpy.sqrt(exposure['mag_r_lsst'])
        error2 = 2.5 / numpy.log(10) * catalog['mag_i_lsst_err'].values * flux2 * numpy.sqrt(exposure['mag_i_lsst'])
        
        mu1 = flux1 / error1
        mu2 = flux2 / error2
        catalog['mu'] = (flux1 * exposure['mag_r_lsst'] / numpy.square(error1) + flux2 * exposure['mag_i_lsst'] / numpy.square(error2)) / numpy.sqrt(exposure['mag_r_lsst'] / numpy.square(error1) + exposure['mag_i_lsst'] / numpy.square(error2))
        
        # Eta
        radius_psf1 = psf_fwhm['psf_r_lsst'] / 2 / numpy.sqrt(numpy.log(2))
        radius_psf2 = psf_fwhm['psf_i_lsst'] / 2 / numpy.sqrt(numpy.log(2))
        
        eta1 = numpy.square(catalog['radius'] / radius_psf1)
        eta2 = numpy.square(catalog['radius'] / radius_psf2)
        catalog['eta'] = (exposure['mag_r_lsst'] / numpy.square(error1) + exposure['mag_i_lsst'] / numpy.square(error2)) * numpy.square(catalog['radius']) / (exposure['mag_r_lsst'] * numpy.square(radius_psf1 / error1) + exposure['mag_i_lsst'] * numpy.square(radius_psf2 / error2))
        
        # Sigma
        a = 1.58
        b = 5.03
        c = 0.39
        
        sigma1 = a / mu1 * (1 + numpy.power(b / eta1, c))
        sigma2 = a / mu2 * (1 + numpy.power(b / eta2, c))
        catalog['sigma'] = 1 / numpy.sqrt(exposure['mag_r_lsst'] / numpy.square(sigma1) + exposure['mag_i_lsst'] / numpy.square(sigma2))
        
        # Condition
        z1 = 0.0
        z2 = 3.0
        condition = (z1 < catalog['redshift_true'].values) & (catalog['redshift_true'].values < z2)
        
        magnitude1 = {'Y1': 16, 'Y10': 16}
        magnitude2 = {'Y1': 24.05, 'Y10': 25.30}
        condition = condition & (magnitude1[tag] < catalog['mag_i_lsst'].values) & (catalog['mag_i_lsst'].values < magnitude2[tag])
        
        mu = 10
        eta = 0.1
        condition = condition & (catalog['mu'].values > mu) & (catalog['eta'].values > eta)
        
        # Append
        for key in catalog.keys():
            if key in simulation.keys():
                simulation[key] = numpy.concatenate([simulation[key], catalog[key][condition]])
            else:
                simulation[key] = catalog[key][condition]
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/SIMULATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        for key in simulation.keys():
            file.create_dataset(key, data=simulation[key], dtype=simulation[key].dtype)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Simulation Datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the simulation datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the simulation datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)