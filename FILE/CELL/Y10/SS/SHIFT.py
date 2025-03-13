import os
import h5py
import json
import time
import numpy
import pyccl
import scipy
import argparse
import multiprocessing
from itertools import product


def cell(cosmology, z_grid, ell_grid, shift_source, bin_source_size):
    '''
    Calculate the shape-shape angular power spectra
    
    Arguments:
        cosmology (pyccl.Cosmology): The cosmology
        z_grid (numpy.ndarray): The redshift grid
        ell_grid (numpy.ndarray): The multipole grid
        shift_source (numpy.ndarray): The source data
        bin_source_size (int): The size of the source bins
    
    Returns:
        cell_data (numpy.ndarray): The shape-shape angular power spectra
    '''
    ell_size = ell_grid.size - 1
    cell_data = numpy.zeros((bin_source_size, bin_source_size, ell_size))
    
    for (m, n) in product(range(bin_source_size), range(bin_source_size)):
        if m <= n:
            tracer1 = pyccl.tracers.WeakLensingTracer(cosmo = cosmology, dndz = [z_grid, shift_source[m, :]], has_shear = True, ia_bias = None, use_A_ia = False, n_samples = z_grid.size)
            tracer2 = pyccl.tracers.WeakLensingTracer(cosmo = cosmology, dndz = [z_grid, shift_source[n, :]], has_shear = True, ia_bias = None, use_A_ia = False, n_samples = z_grid.size)
            cell_grid = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber='auto', limber_max_error=0.001, limber_integration_method='spline', non_limber_integration_method='FKEM', fkem_chi_min=0.001, fkem_Nchi=z_grid.size, p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
            
            cell_value = scipy.interpolate.CubicSpline(x=ell_grid, y=cell_grid, bc_type='natural', extrapolate=True)
            for k in range(ell_size):
                cell_data[m, n, k] = cell_value.integrate(ell_grid[k], ell_grid[k + 1]) / (ell_grid[k + 1] - ell_grid[k])
                cell_data[n, m, k] = cell_data[m, n, k]
    return cell_data


def main(tag, name, label, folder):
    '''
    Calculate the shape-shape angular power spectra
    
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
    
    # Path
    cell_folder = os.path.join(folder, 'CELL/')
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    
    os.makedirs(os.path.join(cell_folder, '{}/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(cell_folder, '{}/{}'.format(tag, name)), exist_ok = True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_size = grid_size + 1
    z_grid = numpy.linspace(z1, z2, z_size)
    
    # Cosmology
    with open(os.path.join(cell_folder, 'COSMOLOGY.json'), 'r') as file:
        cosmology_info = json.load(file)
    
    cosmology = pyccl.Cosmology(
        mass_split = 'single',
        w0=cosmology_info['W0'],
        wa=cosmology_info['WA'],
        n_s=cosmology_info['NS'], 
        A_s=cosmology_info['AS'], 
        m_nu=cosmology_info['M_NU'],
        Neff=cosmology_info['N_EFF'],
        h=cosmology_info['H0'] / 100,
        Omega_k=cosmology_info['OMEGA_K'], 
        Omega_b=cosmology_info['OMEGA_B'], 
        Omega_c=cosmology_info['OMEGA_CDM'],
        Omega_g=cosmology_info['OMEGA_GAMMA'],
        matter_power_spectrum = 'halofit', 
        transfer_function = 'boltzmann_camb',
        extra_parameters = {'camb': {'kmax': 100, 'lmax': 5000, 'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.8}}
    )
    
    pyccl.spline_params.N_K = grid_size
    pyccl.gsl_params.NZ_NORM_SPLINE_INTEGRATION = True
    pyccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = True
    
    # Multipole
    ell1 = 20
    ell2 = 2000
    ell_size = 20
    ell_grid = numpy.geomspace(ell1, ell2, ell_size + 1)
    
    # Size
    select_size = 10000
    
    # SOM
    with h5py.File(os.path.join(analyze_folder, '{}/SOM_{}.hdf5'.format(tag, label)), 'r') as file:
        som_shift_source = file['source']['shift'][...]
    
    data_size, bin_source_size, z_size = som_shift_source.shape
    indices = numpy.random.choice(data_size, select_size, replace = False)
    
    with multiprocessing.Pool(processes = multiprocessing.cpu_count()) as pool:
        som_cell_data = numpy.stack(pool.starmap(cell, [(cosmology, z_grid, ell_grid, som_shift_source[index, :, :], bin_source_size) for index in indices]), axis = 0)
    som_cell_average = numpy.median(som_cell_data, axis = 0)
    
    # Save
    with h5py.File(os.path.join(cell_folder, '{}/{}/SOM_SHIFT_{}.hdf5'.format(tag, name, label)), 'w') as file:
        file.create_dataset('shift', data = som_cell_data)
        file.create_dataset('average', data = som_cell_average)
    del som_shift_source, som_cell_data, som_cell_average
    
    # Model
    with h5py.File(os.path.join(analyze_folder, '{}/MODEL_{}.hdf5'.format(tag, label)), 'r') as file:
        model_shift_source = file['source']['shift'][...]
    
    data_size, bin_source_size, z_size = model_shift_source.shape
    indices = numpy.random.choice(data_size, select_size, replace = False)
    
    with multiprocessing.Pool(processes = multiprocessing.cpu_count()) as pool:
        model_cell_data = numpy.stack(pool.starmap(cell, [(cosmology, z_grid, ell_grid, model_shift_source[index, :, :], bin_source_size) for index in indices]), axis = 0)
    model_cell_average = numpy.median(model_cell_data, axis = 0)
    
    # Save
    with h5py.File(os.path.join(cell_folder, '{}/{}/MODEL_SHIFT_{}.hdf5'.format(tag, name, label)), 'w') as file:
        file.create_dataset('shift', data = model_cell_data)
        file.create_dataset('average', data = model_cell_average)
    del model_shift_source, model_cell_data, model_cell_average
    
    # Product
    with h5py.File(os.path.join(analyze_folder, '{}/PRODUCT_{}.hdf5'.format(tag, label)), 'r') as file:
        product_shift_source = file['source']['shift'][...]
    
    data_size, bin_source_size, z_size = product_shift_source.shape
    indices = numpy.random.choice(data_size, select_size, replace = False)
    
    with multiprocessing.Pool(processes = multiprocessing.cpu_count()) as pool:
        product_cell_data = numpy.stack(pool.starmap(cell, [(cosmology, z_grid, ell_grid, product_shift_source[index, :, :], bin_source_size) for index in indices]), axis = 0)
    product_cell_average = numpy.median(product_cell_data, axis = 0)
    
    # Save
    with h5py.File(os.path.join(cell_folder, '{}/{}/PRODUCT_SHIFT_{}.hdf5'.format(tag, name, label)), 'w') as file:
        file.create_dataset('shift', data = product_cell_data)
        file.create_dataset('average', data = product_cell_average)
    del product_shift_source, product_cell_data, product_cell_average
    
    # Fiducial
    with h5py.File(os.path.join(analyze_folder, '{}/FIDUCIAL_{}.hdf5'.format(tag, label)), 'r') as file:
        fiducial_shift_source = file['source']['shift'][...]
    
    data_size, bin_source_size, z_size = fiducial_shift_source.shape
    indices = numpy.random.choice(data_size, select_size, replace = False)
    
    with multiprocessing.Pool(processes = multiprocessing.cpu_count()) as pool:
        fiducial_cell_data = numpy.stack(pool.starmap(cell, [(cosmology, z_grid, ell_grid, fiducial_shift_source[index, :, :], bin_source_size) for index in indices]), axis = 0)
    fiducial_cell_average = numpy.median(fiducial_cell_data, axis = 0)
    
    # Save
    with h5py.File(os.path.join(cell_folder, '{}/{}/FIDUCIAL_SHIFT_{}.hdf5'.format(tag, name, label)), 'w') as file:
        file.create_dataset('shift', data = fiducial_cell_data)
        file.create_dataset('average', data = fiducial_cell_average)
    del fiducial_shift_source, fiducial_cell_data, fiducial_cell_average
    
    # Histogram
    with h5py.File(os.path.join(analyze_folder, '{}/HISTOGRAM_{}.hdf5'.format(tag, label)), 'r') as file:
        histogram_shift_source = file['source']['shift'][...]
    
    data_size, bin_source_size, z_size = histogram_shift_source.shape
    indices = numpy.random.choice(data_size, select_size, replace = False)
    
    with multiprocessing.Pool(processes = multiprocessing.cpu_count()) as pool:
        histogram_cell_data = numpy.stack(pool.starmap(cell, [(cosmology, z_grid, ell_grid, histogram_shift_source[index, :, :], bin_source_size) for index in indices]), axis = 0)
    histogram_cell_average = numpy.median(histogram_cell_data, axis = 0)
    
    # Save
    with h5py.File(os.path.join(cell_folder, '{}/{}/HISTOGRAM_SHIFT_{}.hdf5'.format(tag, name, label)), 'w') as file:
        file.create_dataset('shift', data = histogram_cell_data)
        file.create_dataset('average', data = histogram_cell_average)
    del histogram_shift_source, histogram_cell_data, histogram_cell_average
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Cell Data')
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