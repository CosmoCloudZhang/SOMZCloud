import os
import h5py
import time
import numpy
import scipy
import argparse
from matplotlib import pyplot


def main(tag, folder):
    '''
    Plot the shear noise as function of the galaxy sizes and brightnesses
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    
    # Path
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    
    # Summarize
    with h5py.File(os.path.join(synthesize_folder, '{}/SOM.hdf5'.format(tag)), 'r') as file:
        som_lens = file['lens']['average'][...]
        som_source = file['source']['average'][...]
    
    with h5py.File(os.path.join(synthesize_folder, '{}/MODEL.hdf5'.format(tag)), 'r') as file:
        model_lens = file['lens']['average'][...]
        model_source = file['source']['average'][...]
    
    with h5py.File(os.path.join(synthesize_folder, '{}/PRODUCT.hdf5'.format(tag)), 'r') as file:
        product_lens = file['lens']['average'][...]
        product_source = file['source']['average'][...]
    
    with h5py.File(os.path.join(synthesize_folder, '{}/HISTOGRAM.hdf5'.format(tag)), 'r') as file:
        histogram_lens = file['lens']['average'][...]
        histogram_source = file['source']['average'][...]
    
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
        plot[m, 0].plot(z_grid, som_lens[m, :], color='darkblue', linewidth=2.0, linestyle='-')
        
        plot[m, 0].plot(z_grid, model_lens[m, :], color='darkgreen', linewidth=2.0, linestyle='-')
        
        plot[m, 0].plot(z_grid, product_lens[m, :], color='darkorange', linewidth=2.0, linestyle='-')
        
        plot[m, 0].plot(z_grid, histogram_lens[m, :], color='black', linewidth=2.0, linestyle='-')
        
        plot[m, 0].set_ylim(0, 12)
        plot[m, 0].set_xlim(numpy.maximum(z1, center_lens[m] - 0.5), numpy.minimum(numpy.maximum(z1, center_lens[m] - 0.5) + 1.0, z2))
        
        plot[m, 0].set_xlabel(r'$z$')
        plot[m, 0].set_ylabel(r'$\phi \left( z \right)$')
        plot[m, 0].text(x=numpy.minimum(numpy.maximum(z1, center_lens[m] - 0.5) + 1.0, z2) - 0.2, y=9.0, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1), fontsize=20)
        
        if m == 0:
            plot[m, 0].set_title(r'$\mathrm{Lens}$')
    
    for m in range(bin_size):
        plot[m, 1].plot(z_grid, som_lens[m + bin_size, :], color='darkblue', linewidth=2.0, linestyle='-')
        
        plot[m, 1].plot(z_grid, model_lens[m + bin_size, :], color='darkgreen', linewidth=2.0, linestyle='-')
        
        plot[m, 1].plot(z_grid, product_lens[m + bin_size, :], color='darkorange', linewidth=2.0, linestyle='-')
        
        plot[m, 1].plot(z_grid, histogram_lens[m + bin_size, :], color='black', linewidth=2.0, linestyle='-')
        
        plot[m, 1].set_ylim(0, 12)
        plot[m, 1].set_xlim(numpy.maximum(z1, center_lens[m + bin_size] - 0.5), numpy.minimum(numpy.maximum(z1, center_lens[m + bin_size] - 0.5) + 1.0, z2))
        
        plot[m, 1].set_xlabel(r'$z$')
        plot[m, 1].set_ylabel(r'$\phi \left( z \right)$')
        plot[m, 1].text(x=numpy.minimum(numpy.maximum(z1, center_lens[m + bin_size] - 0.5) + 1.0, z2) - 0.2, y=9.0, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + bin_size + 1), fontsize=20)
        
        if m == 0:
            plot[m, 1].set_title(r'$\mathrm{Lens}$')
    
    for m in range(bin_size):
        plot[m, 2].plot(z_grid, som_source[m, :], color='darkblue', linewidth=2.0, linestyle='-')
        
        plot[m, 2].plot(z_grid, model_source[m, :], color='darkgreen', linewidth=2.0, linestyle='-')
        
        plot[m, 2].plot(z_grid, product_source[m, :], color='darkorange', linewidth=2.0, linestyle='-')
        
        plot[m, 2].plot(z_grid, histogram_source[m, :], color='black', linewidth=2.0, linestyle='-')
        
        plot[m, 2].set_ylim(0, 8)
        plot[m, 2].set_xlim(numpy.maximum(z1, center_source[m] - 0.5), numpy.minimum(numpy.maximum(z1, center_source[m] - 0.5) + 1.0, z2))
        
        plot[m, 2].set_xlabel(r'$z$')
        plot[m, 2].set_ylabel(r'$\phi \left( z \right)$')
        plot[m, 2].text(x=numpy.minimum(numpy.maximum(z1, center_source[m] - 0.5) + 1.0, z2) - 0.2, y=6.0, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1), fontsize=20)
        
        if m == 0:
            plot[m, 2].set_title(r'$\mathrm{Source}$')
    
    os.makedirs(analyze_folder, exist_ok=True)
    os.makedirs(os.path.join(analyze_folder, '{}/MARGINAL/'.format(tag)), exist_ok=True)
    
    figure.subplots_adjust(wspace=0.2, hspace=0.2)
    figure.savefig(os.path.join(analyze_folder, '{}/MARGINAL/FIGURE.pdf'.format(tag)), format='pdf', bbox_inches='tight')
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Analysis Marginal')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)