import os
import h5py
import json
import time
import numpy
import pyccl
import argparse
from matplotlib import pyplot


def main(tag, name, label, folder):
    '''
    Calculate the two-point chi-square distribution
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
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
    value_folder = os.path.join(folder, 'VALUE/')
    calibrate_folder = os.path.join(folder, 'CALIBRATE/')
    os.makedirs(os.path.join(value_folder, '{}/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(value_folder, '{}/DOUBLE/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(value_folder, '{}/DOUBLE/{}/'.format(tag, name)), exist_ok = True)
    
    # Load
    with h5py.File(os.path.join(calibrate_folder, '{}/CORRECT/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        bin_lens = file['meta']['bin_lens'][...]
        bin_source = file['meta']['bin_source'][...]
        
        data_size = file['meta']['data_size'][...]
        bin_lens_size = file['meta']['bin_lens_size'][...]
        bin_source_size = file['meta']['bin_source_size'][...]
    
    with h5py.File(os.path.join(cell_folder, '{}/TT/SHIFT/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        cell_tt_shift = file['data'][...]
    
    with h5py.File(os.path.join(cell_folder, '{}/TT/SCALE/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        cell_tt_scale = file['data'][...]
    
    with h5py.File(os.path.join(cell_folder, '{}/TT/CORRECT/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        cell_tt_correct = file['data'][...]
        cell_tt_average = file['average'][...]
    
    with h5py.File(os.path.join(cell_folder, '{}/TE/SHIFT/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        cell_te_shift = file['data'][...]
    
    with h5py.File(os.path.join(cell_folder, '{}/TE/SCALE/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        cell_te_scale = file['data'][...]
    
    with h5py.File(os.path.join(cell_folder, '{}/TE/CORRECT/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        cell_te_correct = file['data'][...]
        cell_te_average = file['average'][...]
    
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
    ell_data = numpy.sqrt(ell_grid[+1:] * ell_grid[:-1])
    
    # Covariance
    cell_tt_size = bin_lens_size * (bin_lens_size + 1) // 2 * ell_size
    cell_te_size = bin_lens_size * bin_source_size * ell_size
    
    covariance = numpy.loadtxt(os.path.join(cell_folder, '{}/COVARIANCE/{}/MATRIX.ascii'.format(tag, name)))
    covariance = covariance[0: cell_tt_size + cell_te_size, 0: cell_tt_size + cell_te_size]
    
    # Data
    data_average = numpy.zeros((cell_tt_size + cell_te_size))
    mask = numpy.zeros(cell_tt_size + cell_te_size, dtype=bool)
    data_shift = numpy.zeros((data_size, cell_tt_size + cell_te_size))
    data_scale = numpy.zeros((data_size, cell_tt_size + cell_te_size))
    data_correct = numpy.zeros((data_size, cell_tt_size + cell_te_size))
    
    k_maximal_tt = 0.1 * cosmology_info['H']
    ell_maximal_tt = k_maximal_tt * pyccl.comoving_radial_distance(cosmo=cosmology, a=1 / (1 + bin_lens)) - 1 / 2
    
    k_maximal_te = 0.1 * cosmology_info['H']
    ell_maximal_te = k_maximal_te * pyccl.comoving_radial_distance(cosmo=cosmology, a=1 / (1 + bin_lens)) - 1 / 2
    
    for i in range(bin_lens_size):
        for j in range(bin_lens_size):
            if i == j:
                index = i * bin_lens_size - i * (i - 1) // 2 + (j - i)
                index2 = (index + 1) * ell_size
                index1 = index * ell_size
                
                data_shift[:, index1:index2] = cell_tt_shift[:, i, j, :]
                data_scale[:, index1:index2] = cell_tt_scale[:, i, j, :]
                data_correct[:, index1:index2] = cell_tt_correct[:, i, j, :]
                
                data_average[index1:index2] = cell_tt_average[i, j, :]
                mask[index1:index2] = ell_data < min(ell_maximal_tt[i], ell_maximal_tt[j])
    
    for i in range(bin_lens_size):
        for j in range(bin_source_size):
            if bin_lens[i] + bin_lens[i + 1] / 2 < bin_source[j] + bin_source[j + 1] / 2:
                index = i * bin_source_size + j
                index1 = cell_tt_size + index * ell_size
                index2 = cell_tt_size + (index + 1) * ell_size
                
                data_shift[:, index1:index2] = cell_te_shift[:, i, j, :]
                data_scale[:, index1:index2] = cell_te_scale[:, i, j, :]
                data_correct[:, index1:index2] = cell_te_correct[:, i, j, :]
                
                data_average[index1:index2] = cell_te_average[i, j, :]
                mask[index1:index2] = ell_data < min(ell_maximal_te[i], ell_maximal_te[j])
    
    # Apply mask
    data_shift_mask = data_shift[:, mask]
    data_scale_mask = data_scale[:, mask]
    data_correct_mask = data_correct[:, mask]
    data_average_mask = data_average[mask]
    covariance_mask = covariance[numpy.ix_(mask, mask)]
    
    # Residuals
    delta_shift = data_shift_mask - data_average_mask
    delta_scale = data_scale_mask - data_average_mask
    delta_correct = data_correct_mask - data_average_mask
    
    # Eigenvalue decomposition of masked covariance
    eigenvalues, eigenvectors = numpy.linalg.eigh(covariance_mask)
    keep = eigenvalues > 0
    
    projection = eigenvectors[:, keep].T
    precision = 1.0 / eigenvalues[keep]
    number = int(numpy.sum(keep))
    
    # Projection
    projection_shift = delta_shift @ projection.T
    projection_scale = delta_scale @ projection.T
    projection_correct = delta_correct @ projection.T
    
    # Normalized chi-square
    chi_shift = numpy.sum(numpy.square(projection_shift) * precision, axis=1) / number
    chi_scale = numpy.sum(numpy.square(projection_scale) * precision, axis=1) / number
    chi_correct = numpy.sum(numpy.square(projection_correct) * precision, axis=1) / number
    
    # Save
    with h5py.File(os.path.join(value_folder, '{}/DOUBLE/{}/{}.hdf5'.format(tag, name, label)), 'w') as file:
        file.create_dataset('number', data=number)
        file.create_dataset('chi_shift', data=chi_shift)
        file.create_dataset('chi_scale', data=chi_scale)
        file.create_dataset('chi_correct', data=chi_correct)
    
    # Chi
    chi1 = 0
    chi2 = 5
    chi_size = 30
    chi_bin = numpy.linspace(chi1, chi2, chi_size)
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    figure, plot = pyplot.subplots(nrows=1, ncols=1, figsize=(12, 6))
    
    plot.axvline(x=1.0, color='black', linestyle='--', linewidth=2.5)
    
    plot.hist(chi_shift, bins=chi_bin, color='darkblue', histtype='step', linewidth=2.5, weights=numpy.ones(data_size) / data_size, label=r'$\mathtt{Shift}$')
    
    plot.hist(chi_scale, bins=chi_bin, color='darkorange', histtype='step', linewidth=2.5, weights=numpy.ones(data_size) / data_size, label=r'$\mathtt{Scale}$')
    
    plot.hist(chi_correct, bins=chi_bin, color='darkred', histtype='step', linewidth=2.5, weights=numpy.ones(data_size) / data_size, label=r'$\mathtt{Correct}$')
    
    plot.set_xlim(chi1, chi2)
    plot.legend(fontsize=25, frameon=True, loc='upper right')
    
    plot.set_ylabel(r'$\mathrm{{Fraction}}$', fontsize=25)
    plot.set_xlabel(r'$\chi^2 / N_{\mathrm{dof}}$', fontsize=25)
    
    figure.savefig(os.path.join(value_folder, '{}/DOUBLE/{}/{}.pdf'.format(tag, name, label)), bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Value Double')
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