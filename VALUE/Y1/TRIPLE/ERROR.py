import os
import h5py
import json
import time
import numpy
import pyccl
import argparse
from matplotlib import pyplot


def main(tag, name, label, number, folder):
    '''
    Calculate the position-shape angular power spectra
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the power spectra
        label (str): The label of the configuration
        number (int): Total number of configurations
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Name: {}, Label: {}'.format(name, label))
    
    # Path
    info_folder = os.path.join(folder, 'INFO/')
    cell_folder = os.path.join(folder, 'CELL/')
    model_folder = os.path.join(folder, 'MODEL/')
    
    os.makedirs(os.path.join(cell_folder, '{}/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(cell_folder, '{}/{}'.format(tag, name)), exist_ok = True)
    
    # Load
    with h5py.File(os.path.join(cell_folder, '{}/NN/CORRECT/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        cell_correct_data = file['data'][...]
        cell_correct_average = file['average'][...]
    bin_lens_size, bin_lens_size, ell_size = cell_correct_average.shape
    
    with h5py.File(os.path.join(cell_folder, '{}/NN/SCALE/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        cell_scale_data = file['data'][...]
    
    with h5py.File(os.path.join(cell_folder, '{}/NN/SHIFT/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        cell_shift_data = file['data'][...]
    
    # Multipole
    ell1 = 20
    ell2 = 2000
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
    bin_lens = []
    for index in range(number + 1):
        with h5py.File(os.path.join(model_folder, '{}/TARGET/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            bin_lens.append(file['bin_lens'][...])
    average_bin_lens = numpy.average(numpy.vstack(bin_lens), axis=0)
    
    # Maximum
    k_maximal_lens = 0.1 * cosmology_info['H']
    ell_maximal_lens = k_maximal_lens * pyccl.comoving_radial_distance(cosmo=cosmology, a=1 / (1 + average_bin_lens)) - 1 / 2
    
    # Delta
    cell_correct_delta = numpy.zeros((bin_lens_size, bin_lens_size, ell_size))
    cell_scale_delta = numpy.zeros((bin_lens_size, bin_lens_size, ell_size))
    cell_shift_delta = numpy.zeros((bin_lens_size, bin_lens_size, ell_size))
    
    # Covariance
    cell_range1 = 0
    cell_range2 = bin_lens_size * (bin_lens_size + 1) // 2 * ell_size
    matrix = numpy.loadtxt(os.path.join(cell_folder, '{}/COVARIANCE/MATRIX_{}.ascii'.format(tag, label)), dtype=numpy.float32)
    
    for m in range(bin_lens_size):
        for n in range(m, bin_lens_size):
            cell_correct_delta[m, n, :] = cell_correct_data[m, n, :] - cell_correct_average[m, n, :]
    
    covariance
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Figure
    zeta1 = 2e-3
    zeta2 = 2e+1
    figure, plot = pyplot.subplots(nrows=bin_lens_size, ncols=1, figsize=(12, 30))
    color_list = ['hotpink', 'darkmagenta', 'darkorchid', 'darkblue', 'deepskyblue', 'darkgreen', 'darkgoldenrod', 'darkorange', 'darksalmon', 'darkred']
    
    index = 0
    for m in range(bin_lens_size):
        for n in range(m, bin_lens_size):
            cell_sigma = sigma[index * ell_size: (index + 1) * ell_size]
            index = index + 1
            
            if m == n:
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
        plot[m].set_ylabel(r'$\zeta_{\theta \theta}^{ab} (\ell)$')
        
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