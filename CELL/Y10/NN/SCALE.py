import os
import h5py
import json
import time
import numpy
import pyccl
import scipy
import argparse
from itertools import product


def main(tag, name, label, folder):
    '''
    Calculate the position-position angular power spectra
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the power spectra
        label (str): The label of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Name: {}, Label: {}'.format(name, label))
    
    # Path
    cell_folder = os.path.join(folder, 'CELL/')
    info_folder = os.path.join(folder, 'INFO/')
    calibrate_folder = os.path.join(folder, 'CALIBRATE/')
    os.makedirs(os.path.join(cell_folder, '{}/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(cell_folder, '{}/NN/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(cell_folder, '{}/NN/SCALE/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(cell_folder, '{}/NN/SCALE/{}/'.format(tag, name)), exist_ok = True)
    
    # Load
    with h5py.File(os.path.join(calibrate_folder, '{}/SCALE/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta'].keys()}
        average_lens = file['lens']['average'][...]
        data_lens = file['lens']['data'][...]
    
    # Size
    z_grid = meta['z_grid']
    data_size, bin_lens_size, z_size = data_lens.shape
    
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
        mass_split='single', matter_power_spectrum='halofit', transfer_function='boltzmann_camb',
        extra_parameters={'camb': {'kmax': 50, 'lmax': 5000, 'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.8}}
    )
    
    pyccl.gsl_params['NZ_NORM_SPLINE_INTEGRATION'] = False
    pyccl.gsl_params['LENSING_KERNEL_SPLINE_INTEGRATION'] = False
    
    pyccl.gsl_params['INTEGRATION_GAUSS_KRONROD_POINTS'] = 100
    pyccl.gsl_params['INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS'] = 100
    
    # Multipole
    ell1 = 20
    ell2 = 2000
    ell_size = 20
    ell_grid = numpy.geomspace(ell1, ell2, ell_size + 1)
    
    # Galaxy
    with open(os.path.join(info_folder, 'GALAXY.json'), 'r') as file:
        galaxy_info = json.load(file)
    galaxy_bias = galaxy_info[tag]
    
    # Magnification
    with open(os.path.join(info_folder, 'MAGNIFICATION.json'), 'r') as file:
        magnification_info = json.load(file)
    magnification_bias = magnification_info[tag]
    
    # Cell Average
    cell_average = numpy.zeros((bin_lens_size, bin_lens_size, ell_size))
    for (i, j) in product(range(bin_lens_size), range(bin_lens_size)):
        tracer1 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, average_lens[i, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, numpy.ones(z_size) * magnification_bias[i]], has_rsd=False, n_samples=z_size)
        tracer2 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, average_lens[j, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, numpy.ones(z_size) * magnification_bias[j]], has_rsd=False, n_samples=z_size)
        cell_grid = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.01, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
        
        cell_value = scipy.interpolate.CubicSpline(x=numpy.log(ell_grid), y=ell_grid * cell_grid, bc_type='natural', extrapolate=True)
        for k in range(ell_size):
            cell_average[i, j, k] = cell_value.integrate(numpy.log(ell_grid[k]), numpy.log(ell_grid[k + 1])) / (ell_grid[k + 1] - ell_grid[k])
    
    # Cell Data
    cell_data = numpy.zeros((data_size, bin_lens_size, bin_lens_size, ell_size))
    for n in range(data_size):
        for (i, j) in product(range(bin_lens_size), range(bin_lens_size)):
            if i <= j:
                tracer1 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, data_lens[n, i, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, numpy.ones(z_size) * magnification_bias[i]], has_rsd=False, n_samples=z_size)
                tracer2 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=[z_grid, data_lens[n, j, :]], bias=[z_grid, galaxy_bias], mag_bias=[z_grid, numpy.ones(z_size) * magnification_bias[j]], has_rsd=False, n_samples=z_size)
                cell_grid = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.01, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
                
                cell_value = scipy.interpolate.CubicSpline(x=numpy.log(ell_grid), y=ell_grid * cell_grid, bc_type='natural', extrapolate=True)
                for k in range(ell_size):
                    cell_data[n, i, j, k] = cell_value.integrate(numpy.log(ell_grid[k]), numpy.log(ell_grid[k + 1])) / (ell_grid[k + 1] - ell_grid[k])
                    cell_data[n, j, i, k] = cell_data[n, i, j, k]
    
    # Save
    with h5py.File(os.path.join(cell_folder, '{}/NN/SCALE/{}/{}.hdf5'.format(tag, name, label)), 'w') as file:
        file.create_dataset('data', data = cell_data)
        file.create_dataset('average', data = cell_average)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Cell NN Scale')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the power spectra')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, LABEL, FOLDER)