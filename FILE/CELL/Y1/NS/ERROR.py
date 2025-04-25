import os
import h5py
import json
import time
import numpy
import pyccl
import argparse
from matplotlib import pyplot


def main(tag, name, type, label, folder):
    '''
    Calculate the position-shape angular power spectra
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the power spectra
        type (str): The type of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Type: {}, Label: {}'.format(type, label))
    
    # Path
    info_folder = os.path.join(folder, 'INFO/')
    cell_folder = os.path.join(folder, 'CELL/')
    model_folder = os.path.join(folder, 'MODEL/')
    
    os.makedirs(os.path.join(cell_folder, '{}/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(cell_folder, '{}/{}'.format(tag, name)), exist_ok = True)
    
    # Load
    with h5py.File(os.path.join(cell_folder, '{}/{}/{}_DATA_{}.hdf5'.format(tag, name, type, label)), 'r') as file:
        cell_data = file['data'][...]
    cell_data_error = numpy.std(cell_data, axis = 0)
    
    with h5py.File(os.path.join(cell_folder, '{}/{}/{}_SHIFT_{}.hdf5'.format(tag, name, type, label)), 'r') as file:
        cell_shift = file['data'][...]
    cell_shift_error = numpy.std(cell_shift, axis = 0)
    
    with h5py.File(os.path.join(cell_folder, '{}/{}/{}_SCALE_{}.hdf5'.format(tag, name, type, label)), 'r') as file:
        cell_scale = file['data'][...]
    cell_scale_error = numpy.std(cell_scale, axis = 0)
    
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
    
    bin_lens_size = len(bin_lens) - 1
    bin_source_size = len(bin_source) - 1
    
    k_maximal_lens = 0.1 * cosmology_info['H']
    ell_maximal_lens = k_maximal_lens * pyccl.comoving_radial_distance(cosmo=cosmology, a=1 / (1 + bin_lens)) - 1 / 2
    
    # Covariance
    cell_range1 = bin_lens_size * (bin_lens_size + 1) // 2 * ell_size
    cell_range2 = cell_range1 + bin_lens_size * bin_source_size * ell_size
    
    covariance = numpy.loadtxt(os.path.join(cell_folder, '{}/COVARIANCE/COVARIANCE_MATRIX_{}.ascii'.format(tag, label)), dtype=numpy.float32)
    variance = numpy.diagonal(covariance, axis1=0, axis2=1)
    sigma = numpy.sqrt(variance)[cell_range1: cell_range2]
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Figure
    zeta1 = 5e-4
    zeta2 = 5e-1
    figure, plot = pyplot.subplots(nrows=bin_lens_size, ncols=1, figsize=(12, 30))
    color_list = ['darkmagenta', 'darkblue', 'darkgreen', 'darkorange', 'darkred']
    
    index = 0
    for m in range(bin_lens_size):
        for n in range(bin_source_size):
            cell_sigma = sigma[index * ell_size: (index + 1) * ell_size]
            index = index + 1
            
            if bin_lens[m + 1] < (bin_source[n] + bin_source[n + 1]) / 2:
                cell_shift_zeta = numpy.divide(numpy.abs(cell_shift_error[m, n, :] - cell_data_error[m, n, :]), cell_sigma, out=numpy.zeros(ell_size), where=cell_sigma != 0)
                cell_scale_zeta = numpy.divide(numpy.abs(cell_scale_error[m, n, :] - cell_data_error[m, n, :]), cell_sigma, out=numpy.zeros(ell_size), where=cell_sigma != 0)
                
                plot[m].scatter(ell_data, cell_shift_zeta, s=100, marker='s', facecolors='none', edgecolors=color_list[n])
                plot[m].plot(ell_data, cell_shift_zeta, linestyle='--', linewidth=2.0, color=color_list[n], label=r'${} \times {}$'.format(m + 1, n + 1))
                
                plot[m].plot(ell_data, cell_scale_zeta, linestyle=':', linewidth=2.0, color=color_list[n])
                plot[m].scatter(ell_data, cell_scale_zeta, s=100, marker='d', facecolors='none', edgecolors=color_list[n])
        
        plot[m].axhline(y=1e-0, color='black', linestyle='-.', linewidth=1.0)
        plot[m].axhline(y=1e-1, color='black', linestyle='-.', linewidth=1.0)
        plot[m].axhline(y=1e-2, color='black', linestyle='-.', linewidth=1.0)
        plot[m].fill_betweenx(y=[zeta1, zeta2], x1=ell_maximal_lens[m], x2=ell2, color='gray', alpha=0.2)
        
        plot[m].set_xscale('log')
        plot[m].set_yscale('log')
        plot[m].legend(loc='center left', bbox_to_anchor=(1.0, 0.8), fontsize=25)
        
        plot[m].set_xlim(ell1, ell2)
        plot[m].set_ylim(zeta1, zeta2)
        plot[m].set_ylabel(r'$\zeta_{\theta \epsilon}^{ab} (\ell)$')
        
        if m == bin_lens_size - 1:
            plot[m].set_xlabel(r'$\ell$')
        else:
            plot[m].set_xticklabels([])
    
    figure.subplots_adjust(wspace=0.00, hspace=0.05)
    figure.savefig(os.path.join(cell_folder, '{}/{}/{}_ERROR_{}.pdf'.format(tag, name, type, label)), format='pdf', bbox_inches='tight')
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Cell Error')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the power spectra')
    PARSE.add_argument('--type', type=str, required=True, help='The type of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    TYPE = PARSE.parse_args().type
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, TYPE, LABEL, FOLDER)