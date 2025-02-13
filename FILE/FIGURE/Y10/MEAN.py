import os
import time
import h5py
import numpy
import scipy
import argparse
from matplotlib import pyplot
import scipy.interpolate


def main(tag, folder):
    '''
    Plot the shear noise as function of the galaxy sizes and brightnesses
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    figure_folder = os.path.join(folder, 'FIGURE/')
    ensemble_folder = os.path.join(folder, 'ENSEMBLE/')
    
    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/MEAN/'.format(tag)), exist_ok=True)
    
    # Ensemble
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/FZB.hdf5'.format(tag)), 'r') as file:
        fzb_lens = file['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/SOM.hdf5'.format(tag)), 'r') as file:
        som_lens = file['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/HISTOGRAM.hdf5'.format(tag)), 'r') as file:
        histogram_lens = file['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/FZB.hdf5'.format(tag)), 'r') as file:
        fzb_source = file['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/SOM.hdf5'.format(tag)), 'r') as file:
        som_source = file['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/HISTOGRAM.hdf5'.format(tag)), 'r') as file:
        histogram_source = file['ensemble'][...]
    
    # Bin
    bin_size = 5
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Mean
    fzb_mean_lens = scipy.integrate.trapezoid(x=z_grid, y=fzb_lens * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2)
    som_mean_lens = scipy.integrate.trapezoid(x=z_grid, y=som_lens * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2)
    histogram_mean_lens = scipy.integrate.trapezoid(x=z_grid, y=histogram_lens * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2)
    
    fzb_mean_source = scipy.integrate.trapezoid(x=z_grid, y=fzb_source * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2)
    som_mean_source = scipy.integrate.trapezoid(x=z_grid, y=som_source * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2)
    histogram_mean_source = scipy.integrate.trapezoid(x=z_grid, y=histogram_source * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2)
    
    fzb_center_lens = numpy.median(fzb_mean_lens, axis=0)
    som_center_lens = numpy.median(som_mean_lens, axis=0)
    histogram_center_lens = numpy.median(histogram_mean_lens, axis=0)
    
    fzb_center_source = numpy.median(fzb_mean_source, axis=0)
    som_center_source = numpy.median(som_mean_source, axis=0)
    histogram_center_source = numpy.median(histogram_mean_source, axis=0)
    
    fzb_delta_lens = numpy.abs(fzb_center_lens - histogram_center_lens) / (1 + histogram_center_lens)
    som_delta_lens = numpy.abs(som_center_lens - histogram_center_lens) / (1 + histogram_center_lens)
    
    fzb_delta_source = numpy.abs(fzb_center_source - histogram_center_source) / (1 + histogram_center_source)
    som_delta_source = numpy.abs(som_center_source - histogram_center_source) / (1 + histogram_center_source)
    
    lens_shift = 0.05 * (1 + histogram_center_lens)
    source_shift = 0.08 * (1 + histogram_center_source)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot
    size = 100
    figure, plot = pyplot.subplots(ncols=3, nrows=bin_size, figsize=(18, 15))
    
    for m in range(bin_size):
        
        plot[m, 0].hist(histogram_mean_lens[:, m], bins=size, range=(histogram_center_lens[m] - lens_shift[m], histogram_center_lens[m] + lens_shift[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(fzb_mean_lens[:, m], bins=size, range=(histogram_center_lens[m] - lens_shift[m], histogram_center_lens[m] + lens_shift[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(som_mean_lens[:, m], bins=size, range=(histogram_center_lens[m] - lens_shift[m], histogram_center_lens[m] + lens_shift[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].text(histogram_center_lens[m] - 0.8 * lens_shift[m], 120, r'$\delta_{\langle z \rangle}^\mathrm{FZB} =$' + r'${:.3f}$'.format(fzb_delta_lens[m]), color='darkorange', fontsize=20)
        
        plot[m, 0].text(histogram_center_lens[m] + 0.2 * lens_shift[m], 120, r'$\delta_{\langle z \rangle}^\mathrm{SOM} =$' + r'${:.3f}$'.format(som_delta_lens[m]), color='darkgreen', fontsize=20)
        
        plot[m, 0].set_ylim(0, 150)
        plot[m, 0].set_yticklabels([])
        plot[m, 0].set_xlim(histogram_center_lens[m] - lens_shift[m], histogram_center_lens[m] + lens_shift[m])
        
        if m == bin_size - 1: 
            plot[m, 0].set_xlabel(r'$\langle z \rangle$')
        plot[m, 0].set_ylabel(r'$\mathcal{P} \left( \langle z \rangle \right)$')
        
        plot[m, 1].hist(histogram_mean_lens[:, m + bin_size], bins=size, range=(histogram_center_lens[m + bin_size] - lens_shift[m + bin_size], histogram_center_lens[m + bin_size] + lens_shift[m + bin_size]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(fzb_mean_lens[:, m + bin_size], bins=size, range=(histogram_center_lens[m + bin_size] - lens_shift[m + bin_size], histogram_center_lens[m + bin_size] + lens_shift[m + bin_size]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(som_mean_lens[:, m + bin_size], bins=size, range=(histogram_center_lens[m + bin_size] - lens_shift[m + bin_size], histogram_center_lens[m + bin_size] + lens_shift[m + bin_size]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].text(histogram_center_lens[m + bin_size] - 0.8 * lens_shift[m + bin_size], 100, r'$\delta_{\langle z \rangle}^\mathrm{FZB} =$' + r'${:.3f}$'.format(fzb_delta_lens[m + bin_size]), color='darkorange', fontsize=20)
        
        plot[m, 1].text(histogram_center_lens[m + bin_size] + 0.2 * lens_shift[m + bin_size], 100, r'$\delta_{\langle z \rangle}^\mathrm{SOM} =$' + r'${:.3f}$'.format(som_delta_lens[m + bin_size]), color='darkgreen', fontsize=20)
        
        plot[m, 1].set_ylim(0, 150)
        plot[m, 1].set_yticklabels([])
        plot[m, 1].set_xlim(histogram_center_lens[m + bin_size] - lens_shift[m + bin_size], histogram_center_lens[m + bin_size] + lens_shift[m + bin_size])
        
        if m == bin_size - 1:
            plot[m, 1].set_xlabel(r'$\langle z \rangle$')
        
        plot[m, 2].hist(histogram_mean_source[:, m], bins=size, range=(histogram_center_source[m] - source_shift[m], histogram_center_source[m] + source_shift[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 2].hist(fzb_mean_source[:, m], bins=size, range=(histogram_center_source[m] - source_shift[m], histogram_center_source[m] + source_shift[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 2].hist(som_mean_source[:, m], bins=size, range=(histogram_center_source[m] - source_shift[m], histogram_center_source[m] + source_shift[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 2].text(histogram_center_source[m] - 0.8 * source_shift[m], 120, r'$\delta_{\langle z \rangle}^\mathrm{FZB} =$' + r'${:.3f}$'.format(fzb_delta_source[m]), color='darkorange', fontsize=20)
        
        plot[m, 2].text(histogram_center_source[m] + 0.2 * source_shift[m], 120, r'$\delta_{\langle z \rangle}^\mathrm{SOM} =$' + r'${:.3f}$'.format(som_delta_source[m]), color='darkgreen', fontsize=20)
        
        plot[m, 2].set_ylim(0, 150)
        plot[m, 2].set_yticklabels([])
        plot[m, 2].set_xlim(histogram_center_source[m] - source_shift[m], histogram_center_source[m] + source_shift[m])
        
        if m == bin_size - 1:
            plot[m, 2].set_xlabel(r'$\langle z \rangle$')
        
    figure.subplots_adjust(wspace=0.0, hspace=0.2)
    figure.savefig(os.path.join(figure_folder, '{}/MEAN/FIGURE.png'.format(tag)), bbox_inches='tight', format='png', dpi=512)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Mean')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)
