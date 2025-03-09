import os
import time
import h5py
import numpy
import scipy
import argparse
from matplotlib import pyplot
import scipy.integrate


def main(tag, index, folder):
    '''
    Plot the average conditional ensemble redshift distribution
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    figure_folder = os.path.join(folder, 'FIGURE/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    
    # Summarize Lens
    with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/SOM.hdf5'.format(tag, index)), 'r') as file:
        som_lens = file['average'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/MODEL.hdf5'.format(tag, index)), 'r') as file:
        model_lens = file['average'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/PRODUCT.hdf5'.format(tag, index)), 'r') as file:
        product_lens = file['average'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/HISTOGRAM.hdf5'.format(tag, index)), 'r') as file:
        histogram_lens = file['average'][...]
    
    # Summarize Source
    with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/SOM.hdf5'.format(tag, index)), 'r') as file:
        som_source = file['average'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/MODEL.hdf5'.format(tag, index)), 'r') as file:
        model_source = file['average'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/PRODUCT.hdf5'.format(tag, index)), 'r') as file:
        product_source = file['average'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/HISTOGRAM.hdf5'.format(tag, index)), 'r') as file:
        histogram_source = file['average'][...]
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * histogram_lens, axis=1)
    center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * histogram_source, axis=1)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot
    bin_size = 5
    figure, plot = pyplot.subplots(nrows=bin_size, ncols=3, figsize=(18, 3 * bin_size))
    
    for m in range(bin_size):
        plot[m, 0].plot(z_grid, som_lens[m, :], color='darkblue', linewidth=1.5, linestyle='-')
        
        plot[m, 0].plot(z_grid, model_lens[m, :], color='darkgreen', linewidth=1.5, linestyle='-')
        
        plot[m, 0].plot(z_grid, product_lens[m, :], color='darkorange', linewidth=1.5, linestyle='-')
        
        plot[m, 0].plot(z_grid, histogram_lens[m, :], color='black', linewidth=1.5, linestyle='-')
        
        plot[m, 0].set_ylim(0, 12)
        plot[m, 0].set_xlim(numpy.maximum(z1, center_lens[m] - 0.5), numpy.minimum(numpy.maximum(z1, center_lens[m] - 0.5) + 1.0, z2))
        
        plot[m, 0].set_xlabel(r'$z$')
        plot[m, 0].set_ylabel(r'$\phi \left( z \right)$')
        plot[m, 0].text(x=numpy.minimum(numpy.maximum(z1, center_lens[m] - 0.5) + 1.0, z2) - 0.2, y=10.0, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1), fontsize=20)
        
        if m == 0:
            plot[m, 0].set_title(r'$\mathrm{Lens}$')
    
    for m in range(bin_size):
        plot[m, 1].plot(z_grid, som_lens[m + bin_size, :], color='darkblue', linewidth=1.5, linestyle='-')
        
        plot[m, 1].plot(z_grid, model_lens[m + bin_size, :], color='darkgreen', linewidth=1.5, linestyle='-')
        
        plot[m, 1].plot(z_grid, product_lens[m + bin_size, :], color='darkorange', linewidth=1.5, linestyle='-')
        
        plot[m, 1].plot(z_grid, histogram_lens[m + bin_size, :], color='black', linewidth=1.5, linestyle='-')
        
        plot[m, 1].set_ylim(0, 12)
        plot[m, 1].set_xlim(numpy.maximum(z1, center_lens[m + bin_size] - 0.5), numpy.minimum(numpy.maximum(z1, center_lens[m + bin_size] - 0.5) + 1.0, z2))
        
        plot[m, 1].set_xlabel(r'$z$')
        plot[m, 1].set_ylabel(r'$\phi \left( z \right)$')
        plot[m, 1].text(x=numpy.minimum(numpy.maximum(z1, center_lens[m + bin_size] - 0.5) + 1.0, z2) - 0.2, y=10.0, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + bin_size + 1), fontsize=20)
        
        if m == 0:
            plot[m, 1].set_title(r'$\mathrm{Lens}$')
    
    for m in range(bin_size):
        plot[m, 2].plot(z_grid, som_source[m, :], color='darkblue', linewidth=1.5, linestyle='-')
        
        plot[m, 2].plot(z_grid, model_source[m, :], color='darkgreen', linewidth=1.5, linestyle='-')
        
        plot[m, 2].plot(z_grid, product_source[m, :], color='darkorange', linewidth=1.5, linestyle='-')
        
        plot[m, 2].plot(z_grid, histogram_source[m, :], color='black', linewidth=1.5, linestyle='-')
        
        plot[m, 2].set_ylim(0, 6)
        plot[m, 2].set_xlim(numpy.maximum(z1, center_source[m] - 0.5), numpy.minimum(numpy.maximum(z1, center_source[m] - 0.5) + 1.0, z2))
        
        plot[m, 2].set_xlabel(r'$z$')
        plot[m, 2].set_ylabel(r'$\phi \left( z \right)$')
        plot[m, 2].text(x=numpy.minimum(numpy.maximum(z1, center_source[m] - 0.5) + 1.0, z2) - 0.2, y=4.0, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1), fontsize=20)
        
        if m == 0:
            plot[m, 2].set_title(r'$\mathrm{Source}$')
    
    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/CONDITION/'.format(tag)), exist_ok=True)
    
    figure.subplots_adjust(wspace=0.2, hspace=0.2)
    figure.savefig(os.path.join(figure_folder, '{}/CONDITION/FIGURE{}.pdf'.format(tag, index)), format='pdf', bbox_inches='tight')
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Condition')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)