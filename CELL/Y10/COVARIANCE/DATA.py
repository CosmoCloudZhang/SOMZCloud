import os
import h5py
import json
import time
import numpy
import pyccl
import argparse
from astropy import table
from itertools import product


def main(tag, name, folder):
    '''
    Calculate information for covariance matrix of angular power spectra
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Name: {}'.format(name))
    
    # Path
    cell_folder = os.path.join(folder, 'CELL/')
    info_folder = os.path.join(folder, 'INFO/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    os.makedirs(os.path.join(cell_folder, '{}/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(cell_folder, '{}/COVARIANCE/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(cell_folder, '{}/COVARIANCE/{}'.format(tag, name)), exist_ok = True)
    
    # Load
    with h5py.File(os.path.join(synthesize_folder, '{}/{}/TRUTH.hdf5'.format(tag, name)), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta'].keys()}
        truth_average_source = file['source']['average'][...]
        truth_average_lens = file['lens']['average'][...]
    
    # Meta
    z_grid = meta['z_grid']
    grid_size = int(meta['grid_size'][...])
    bin_lens_size = int(meta['bin_lens_size'][...])
    bin_source_size = int(meta['bin_source_size'][...])
    
    # Lens
    table_lens = table.Table()
    table_lens['redshift'] = z_grid
    for m in range(bin_lens_size):
        table_lens['n_{}(z)'.format(m + 1)] = truth_average_lens[m, :]
    table_lens.write(os.path.join(cell_folder, '{}/COVARIANCE/{}/LENS.ascii'.format(tag, name)), overwrite = True, format = 'ascii.commented_header')
    
    # Source
    table_source = table.Table()
    table_source['redshift'] = z_grid
    for m in range(bin_source_size):
        table_source['n_{}(z)'.format(m + 1)] = truth_average_source[m, :]
    table_source.write(os.path.join(cell_folder, '{}/COVARIANCE/{}/SOURCE.ascii'.format(tag, name)), overwrite = True, format = 'ascii.commented_header')
    
    # Alignment
    with open(os.path.join(info_folder, 'ALIGNMENT.json'), 'r') as file:
        alignment_info = json.load(file)
    alignment_bias = numpy.array(alignment_info['A'])
    
    table_alignment = table.Table()
    table_alignment['redshift'] = z_grid
    for m in range(bin_source_size):
        table_alignment['A_{}(z)'.format(m + 1)] = alignment_bias
    table_alignment.write(os.path.join(cell_folder, '{}/COVARIANCE/{}/ALIGNMENT.ascii'.format(tag, name)), overwrite = True, format = 'ascii.commented_header')
    
    # Galaxy
    with open(os.path.join(info_folder, 'GALAXY.json'), 'r') as file:
        galaxy_info = json.load(file)
    galaxy_bias = numpy.array(galaxy_info[tag])
    
    table_galaxy = table.Table()
    table_galaxy['redshift'] = z_grid
    for m in range(bin_lens_size):
        table_galaxy['b_{}(z)'.format(m + 1)] = galaxy_bias
    table_galaxy.write(os.path.join(cell_folder, '{}/COVARIANCE/{}/GALAXY.ascii'.format(tag, name)), overwrite = True, format = 'ascii.commented_header')
    
    # Magnification
    with open(os.path.join(info_folder, 'MAGNIFICATION.json'), 'r') as file:
        magnification_info = json.load(file)
    magnification_bias = numpy.array(magnification_info[tag])
    
    table_magnification = table.Table()
    table_magnification['redshift'] = z_grid
    for m in range(bin_source_size):
        table_magnification['m_{}(z)'.format(m + 1)] = magnification_bias[m] * numpy.ones(grid_size + 1)
    table_magnification.write(os.path.join(cell_folder, '{}/COVARIANCE/{}/MAGNIFICATION.ascii'.format(tag, name)), overwrite = True, format = 'ascii.commented_header')
    
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
        Omega_b=cosmology_info['OMEGA_B'], 
        Omega_k=cosmology_info['OMEGA_K'], 
        Omega_c=cosmology_info['OMEGA_CDM'], 
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
    ell_size = 1000
    ell_grid = numpy.geomspace(ell1, ell2, ell_size + 1)
    
    # Cell TT
    ell = numpy.zeros((bin_lens_size, bin_lens_size, ell_size + 1), dtype=numpy.float32)
    index1 = numpy.zeros((bin_lens_size, bin_lens_size, ell_size + 1), dtype=numpy.int32)
    index2 = numpy.zeros((bin_lens_size, bin_lens_size, ell_size + 1), dtype=numpy.int32)
    value = numpy.zeros((bin_lens_size, bin_lens_size, ell_size + 1), dtype=numpy.float32)
    
    for (i, j) in product(range(bin_lens_size), range(bin_lens_size)):
        tracer1 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=(z_grid, truth_average_lens[i, :]), bias=(z_grid, galaxy_bias), mag_bias=(z_grid, magnification_bias[i] * numpy.ones(grid_size + 1)), has_rsd=False, n_samples=grid_size + 1)
        tracer2 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=(z_grid, truth_average_lens[j, :]), bias=(z_grid, galaxy_bias), mag_bias=(z_grid, magnification_bias[j] * numpy.ones(grid_size + 1)), has_rsd=False, n_samples=grid_size + 1)
        value[i, j, :] = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.001, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
        
        index1[i, j, :] = i + 1
        index2[i, j, :] = j + 1
        ell[i, j, :] = ell_grid
    
    table_cell = table.Table()
    order = numpy.argsort(ell.flatten())
    table_cell['ell'] = ell.flatten()[order]
    table_cell['tomo_i'] = index1.flatten()[order]
    table_cell['tomo_j'] = index2.flatten()[order]
    table_cell['Cell_gg'] = value.flatten()[order]
    table_cell.write(os.path.join(cell_folder, '{}/COVARIANCE/{}/Cell_gg.ascii'.format(tag, name)), overwrite = True, format = 'ascii.commented_header')
    
    # Cell TE
    ell = numpy.zeros((bin_lens_size, bin_source_size, ell_size + 1), dtype=numpy.float32)
    index1 = numpy.zeros((bin_lens_size, bin_source_size, ell_size + 1), dtype=numpy.int32)
    index2 = numpy.zeros((bin_lens_size, bin_source_size, ell_size + 1), dtype=numpy.int32)
    value = numpy.zeros((bin_lens_size, bin_source_size, ell_size + 1), dtype=numpy.float32)
    
    for (i, j) in product(range(bin_lens_size), range(bin_source_size)):
        tracer1 = pyccl.tracers.NumberCountsTracer(cosmo=cosmology, dndz=(z_grid, truth_average_lens[i, :]), bias=(z_grid, galaxy_bias), mag_bias=(z_grid, magnification_bias[i] * numpy.ones(grid_size + 1)), has_rsd=False, n_samples=grid_size + 1)
        tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=(z_grid, truth_average_source[j, :]), has_shear=True, ia_bias=(z_grid, alignment_bias), use_A_ia=False, n_samples=grid_size + 1)
        value[i, j, :] = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.001, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
        
        index1[i, j, :] = i + 1
        index2[i, j, :] = j + 1
        ell[i, j, :] = ell_grid
    
    table_cell = table.Table()
    order = numpy.argsort(ell.flatten())
    table_cell['ell'] = ell.flatten()[order]
    table_cell['tomo_i'] = index1.flatten()[order]
    table_cell['tomo_j'] = index2.flatten()[order]
    table_cell['Cell_gkappa'] = value.flatten()[order]
    table_cell.write(os.path.join(cell_folder, '{}/COVARIANCE/{}/Cell_gkappa.ascii'.format(tag, name)), overwrite = True, format = 'ascii.commented_header')
    
    # Cell EE
    ell = numpy.zeros((bin_source_size, bin_source_size, ell_size + 1), dtype=numpy.float32)
    index1 = numpy.zeros((bin_source_size, bin_source_size, ell_size + 1), dtype=numpy.int32)
    index2 = numpy.zeros((bin_source_size, bin_source_size, ell_size + 1), dtype=numpy.int32)
    value = numpy.zeros((bin_source_size, bin_source_size, ell_size + 1), dtype=numpy.float32)
    
    for (i, j) in product(range(bin_source_size), range(bin_source_size)):
        tracer1 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=(z_grid, truth_average_source[i, :]), has_shear=True, ia_bias=(z_grid, alignment_bias), use_A_ia=False, n_samples=grid_size + 1)
        tracer2 = pyccl.tracers.WeakLensingTracer(cosmo=cosmology, dndz=(z_grid, truth_average_source[j, :]), has_shear=True, ia_bias=(z_grid, alignment_bias), use_A_ia=False, n_samples=grid_size + 1)
        value[i, j, :] = pyccl.cells.angular_cl(cosmo=cosmology, tracer1=tracer1, tracer2=tracer2, ell=ell_grid, p_of_k_a='delta_matter:delta_matter', l_limber=-1, limber_max_error=0.001, limber_integration_method='spline', p_of_k_a_lin='delta_matter:delta_matter', return_meta=False)
        
        index1[i, j, :] = i + 1
        index2[i, j, :] = j + 1
        ell[i, j, :] = ell_grid
    
    table_cell = table.Table()
    order = numpy.argsort(ell.flatten())
    table_cell['ell'] = ell.flatten()[order]
    table_cell['tomo_i'] = index1.flatten()[order]
    table_cell['tomo_j'] = index2.flatten()[order]
    table_cell['Cell_kappakappa'] = value.flatten()[order]
    table_cell.write(os.path.join(cell_folder, '{}/COVARIANCE/{}/Cell_kappakappa.ascii'.format(tag, name)), overwrite = True, format = 'ascii.commented_header')
    
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
    PARSE.add_argument('--name', type=str, required=True, help='The name of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, FOLDER)