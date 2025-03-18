import os
import h5py
import json
import time
import numpy
import pyccl
import argparse
from matplotlib import cm, pyplot


def main(tag, name, type, label, folder):
    '''
    Calculate the shape-shape angular power spectra
    
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
        bin_source = file['bin_source'][...]
    k_maximal = 10 * cosmology_info['H']
    ell_maximal = k_maximal * pyccl.comoving_radial_distance(cosmo=cosmology, a=1 / (1 + bin_source)) - 1 / 2
    
    # Load
    with h5py.File(os.path.join(cell_folder, '{}/{}/{}_DATA_{}.hdf5'.format(tag, name, type, label)), 'r') as file:
        cell_data = file['data'][...]
    cell_data_error = numpy.std(cell_data, axis = 0)
    
    with h5py.File(os.path.join(cell_folder, '{}/{}/{}_SHIFT_{}.hdf5'.format(tag, name, type, label)), 'r') as file:
        cell_shift = file['data'][...]
    cell_shift_error = numpy.std(cell_shift, axis = 0)
    
    bin_source_size, bin_source_size, ell_size = cell_shift_error.shape
    bin_source_size, bin_source_size, ell_size = cell_data_error.shape
    cell_size = bin_source_size * bin_source_size
    
    # Multipole
    ell1 = 20
    ell2 = 2000
    ell_size = 20
    ell_grid = numpy.geomspace(ell1, ell2, ell_size + 1)
    ell_data = numpy.sqrt(ell_grid[+1:] * ell_grid[:-1])
    
    # Covariance
    covariance_matrix = numpy.loadtxt(os.path.join(cell_folder, '{}/COVARIANCE/COVARIANCE_MATRIX_{}.ascii'.format(tag, label)))
    covariance_matrix = covariance_matrix[:cell_size * ell_size, :cell_size * ell_size]
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Figure
    color_list = cm.rainbow(numpy.linspace(0, 1, cell_size))
    figure, plot = pyplot.subplots(nrows=bin_source_size, ncols=1, figsize=(12, 4 * bin_source_size))
    
    index = 0
    for m in range(bin_source_size):
        for n in range(m, bin_source_size):
            variance = numpy.diag(covariance_matrix[m * ell_size: (m + 1) * ell_size, n * ell_size: (n + 1) * ell_size])
            cell_data_varsigma = numpy.divide(cell_data_error[m, n, :], numpy.sqrt(variance), out=numpy.zeros(ell_size), where=variance != 0)
            cell_shift_varsigma = numpy.divide(cell_shift_error[m, n, :], numpy.sqrt(variance), out=numpy.zeros(ell_size), where=variance != 0)
            
            plot[m].scatter(ell_data, cell_data_varsigma, s=50, marker='s', facecolors=color_list[index], edgecolors=color_list[index])
            plot[m].plot(ell_data, cell_data_varsigma, linestyle='-', linewidth=2.5, color=color_list[index], label=r'$n = {:.0f}$'.format(n + 1))
            
            plot[m].plot(ell_data, cell_shift_varsigma, linestyle=':', linewidth=2.5, color=color_list[index])
            plot[m].scatter(ell_data, cell_shift_varsigma, s=50, marker='s', facecolors='none', edgecolors=color_list[index])
            
            plot[m].axhline(y=1, color='black', linestyle='-')
            plot[m].text(x=1000, y=1e-2, s=r'$m = {:.0f}$'.format(m + 1), fontsize=20, color='black')
            plot[m].fill_betweenx(y=[1e-3, 1e+2], x1=ell_maximal[m], x2=ell2, color='gray', alpha=0.2)
            
            index = index + 1
        
        plot[m].set_xscale('log')
        plot[m].set_yscale('log')
        plot[m].legend(loc='upper left', fontsize=20)
        
        plot[m].set_xlabel(r'$\ell$')
        plot[m].set_ylabel(r'$\varsigma_{\epsilon \epsilon}^{mn} (\ell)$')
        
        plot[m].set_xlim(ell1, ell2)
        plot[m].set_ylim(1e-3, 1e+2)
    
    figure.subplots_adjust(wspace=0, hspace=0)
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