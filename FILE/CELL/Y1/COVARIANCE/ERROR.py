import os
import h5py
import json
import time
import numpy
import pyccl
import argparse
from astropy import table


def main(tag, type, label, folder):
    '''
    Calculate the shape-shape angular power spectra
    
    Arguments:
        tag (str): The tag of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    info_folder = os.path.join(folder, 'INFO/')
    cell_folder = os.path.join(folder, 'CELL/')
    model_folder = os.path.join(folder, 'MODEL/')
    
    # NN
    with h5py.File(os.path.join(cell_folder, '{}/NN/{}_DATA_{}.hdf5'.format(tag, type, label)), 'r') as file:
        cell_nn_data = file['data'][...]
    data_size, bin_lens_size, bin_lens_size, ell_size = cell_nn_data.shape
    
    with h5py.File(os.path.join(cell_folder, '{}/NN/{}_SHIFT_{}.hdf5'.format(tag, type, label)), 'r') as file:
        cell_nn_shift = file['data'][...]
    data_size, bin_lens_size, bin_lens_size, ell_size = cell_nn_shift.shape
    
    # NS
    with h5py.File(os.path.join(cell_folder, '{}/NS/{}_DATA_{}.hdf5'.format(tag, type, label)), 'r') as file:
        cell_ns_data = file['data'][...]
    data_size, bin_lens_size, bin_source_size, ell_size = cell_ns_data.shape
    
    with h5py.File(os.path.join(cell_folder, '{}/NS/{}_SHIFT_{}.hdf5'.format(tag, type, label)), 'r') as file:
        cell_ns_shift = file['data'][...]
    data_size, bin_lens_size, bin_source_size, ell_size = cell_ns_shift.shape
    
    # SS
    with h5py.File(os.path.join(cell_folder, '{}/SS/{}_DATA_{}.hdf5'.format(tag, type, label)), 'r') as file:
        cell_ss_data = file['data'][...]
    data_size, bin_source_size, bin_source_size, ell_size = cell_ss_data.shape
    
    with h5py.File(os.path.join(cell_folder, '{}/SS/{}_SHIFT_{}.hdf5'.format(tag, type, label)), 'r') as file:
        cell_ss_shift = file['data'][...]
    data_size, bin_source_size, bin_source_size, ell_size = cell_ss_shift.shape
    
    cell_ns_size = bin_lens_size * bin_source_size
    cell_nn_size = bin_lens_size * (bin_lens_size + 1) // 2
    cell_ss_size = bin_source_size * (bin_source_size + 1) // 2
    cell_size = (cell_nn_size + cell_ns_size + cell_ss_size) * ell_size
    print(cell_size)
    # Multipole
    ell1 = 20
    ell2 = 2000
    ell_size = 20
    ell_grid = numpy.geomspace(ell1, ell2, ell_size + 1)
    ell_data = numpy.sqrt(ell_grid[+1:] * ell_grid[:-1])
    
    # Cosmology
    with open(os.path.join(info_folder, 'COSMOLOGY.json'), 'r') as file:
        cosmology_info = json.load(file)
    
    cosmology = pyccl.Cosmology(
        h=cosmology_info['H'],
        w0=cosmology_info['W0'],
        wa=cosmology_info['WA'],
        n_s=cosmology_info['NS'], 
        A_s=cosmology_info['AS'], 
        m_nu=cosmology_info['M_NU'],
        Neff=cosmology_info['N_EFF'],
        Omega_k=cosmology_info['OMEGA_K'], 
        Omega_b=cosmology_info['OMEGA_B'], 
        Omega_c=cosmology_info['OMEGA_CDM'],
        Omega_g=cosmology_info['OMEGA_GAMMA'], 
        mass_split = 'single', matter_power_spectrum = 'halofit', transfer_function = 'boltzmann_camb',
        extra_parameters = {'camb': {'kmax': 100, 'lmax': 5000, 'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.8}}
    )
    
    # Bin
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA0.hdf5'.format(tag)), 'r') as file:
        bin_lens = file['bin_lens'][...]
        bin_source = file['bin_source'][...]
    
    k_lens_maximal = 0.1 * cosmology_info['H']
    k_source_maximal = 10 * cosmology_info['H']
    
    ell_lens_maximal = k_lens_maximal * pyccl.comoving_radial_distance(cosmo=cosmology, a=1 / (1 + bin_lens)) - 1 / 2
    ell_source_maximal = k_source_maximal * pyccl.comoving_radial_distance(cosmo=cosmology, a=1 / (1 + bin_source)) - 1 / 2
    
    # Covariance
    covariance = numpy.loadtxt(os.path.join(cell_folder, '{}/COVARIANCE/COVARIANCE_MATRIX_{}.ascii'.format(tag, label)), dtype=numpy.float32)
    matrix = numpy.linalg.inv(covariance)
    print(matrix.shape)
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Cell Covariance')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--type', type=str, required=True, help='The type of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    TYPE = PARSE.parse_args().type
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, TYPE, LABEL, FOLDER)