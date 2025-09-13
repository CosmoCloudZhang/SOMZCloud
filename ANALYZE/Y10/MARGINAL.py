import os
import h5py
import time
import numpy
import scipy
import argparse
from matplotlib import pyplot


def main(tag, name, folder):
    '''
    Plot the marginal distribution of the lens and source galaxies
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    print('Name: {}'.format(name))
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    os.makedirs(os.path.join(analyze_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(analyze_folder, '{}/{}/'.format(tag, name)), exist_ok=True)
    os.makedirs(os.path.join(analyze_folder, '{}/{}/MARGINAL/'.format(tag, name)), exist_ok=True)
    
    # Bin
    with h5py.File(os.path.join(model_folder, '{}/TARGET/DATA0.hdf5'.format(tag)), 'r') as file:
        bin_lens = file['bin_lens'][...]
        bin_source = file['bin_source'][...]
    
    # Summarize
    with h5py.File(os.path.join(synthesize_folder, '{}/{}/DIR.hdf5'.format(tag, name)), 'r') as file:
        dir_average_lens = file['lens']['average'][...]
        dir_average_source = file['source']['average'][...]
    
    with h5py.File(os.path.join(synthesize_folder, '{}/{}/STACK.hdf5'.format(tag, name)), 'r') as file:
        stack_average_lens = file['lens']['average'][...]
        stack_average_source = file['source']['average'][...]
    
    with h5py.File(os.path.join(synthesize_folder, '{}/{}/PRODUCT.hdf5'.format(tag, name)), 'r') as file:
        product_average_lens = file['lens']['average'][...]
        product_average_source = file['source']['average'][...]
    
    with h5py.File(os.path.join(synthesize_folder, '{}/{}/TRUTH.hdf5'.format(tag, name)), 'r') as file:
        truth_average_lens = file['lens']['average'][...]
        truth_average_source = file['source']['average'][...]
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * truth_average_lens, axis=1)
    center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * truth_average_source, axis=1)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 30
    
    # Plot
    bin_size = 5
    range_lens = 0.6
    range_source = [2.0, 2.1, 2.2, 2.3, 2.4]
    figure, plot = pyplot.subplots(nrows=bin_size, ncols=3, figsize=(18, 5 * bin_size))
    
    for m in range(bin_size):
        plot[m, 0].plot(z_grid, dir_average_lens[m, :], color='darkblue', linewidth=2.0, linestyle='-')
        
        plot[m, 0].plot(z_grid, stack_average_lens[m, :], color='darkgreen', linewidth=2.0, linestyle='-')
        
        plot[m, 0].plot(z_grid, product_average_lens[m, :], color='darkorange', linewidth=2.0, linestyle='-')
        
        plot[m, 0].plot(z_grid, truth_average_lens[m, :], color='black', linewidth=2.0, linestyle='-')
        
        plot[m, 0].fill_betweenx(y=[0, 12], x1=bin_lens[m], x2=bin_lens[m + 1], color='gray', alpha=0.5)
        
        plot[m, 0].set_ylim(0, 12)
        plot[m, 0].set_xlim(numpy.maximum(z1, center_lens[m] - range_lens / 2), numpy.minimum(numpy.maximum(z1, center_lens[m] - range_lens / 2) + range_lens, z2))
        
        plot[m, 0].set_yticks([3, 6, 9, 12])
        plot[m, 0].set_ylabel(r'$\phi \left( z \right)$')
        plot[m, 0].text(x=numpy.minimum(numpy.maximum(z1, center_lens[m] - range_lens / 2) + range_lens, z2) - range_lens / 3, y=9.0, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1))
        
        if m == 0:
            plot[m, 0].set_title(r'$\mathrm{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 0].set_xlabel(r'$z$')
    
    for m in range(bin_size):
        plot[m, 1].plot(z_grid, dir_average_lens[m + bin_size, :], color='darkblue', linewidth=2.0, linestyle='-')
        
        plot[m, 1].plot(z_grid, stack_average_lens[m + bin_size, :], color='darkgreen', linewidth=2.0, linestyle='-')
        
        plot[m, 1].plot(z_grid, product_average_lens[m + bin_size, :], color='darkorange', linewidth=2.0, linestyle='-')
        
        plot[m, 1].plot(z_grid, truth_average_lens[m + bin_size, :], color='black', linewidth=2.0, linestyle='-')
        
        plot[m, 1].fill_betweenx(y=[0, 12], x1=bin_lens[m + bin_size], x2=bin_lens[m + bin_size + 1], color='gray', alpha=0.5)
        
        plot[m, 1].set_ylim(0, 12)
        plot[m, 1].set_xlim(numpy.maximum(z1, center_lens[m + bin_size] - range_lens / 2), numpy.minimum(numpy.maximum(z1, center_lens[m + bin_size] - range_lens / 2) + range_lens, z2))
        
        plot[m, 1].set_yticks([3, 6, 9, 12])
        plot[m, 1].text(x=numpy.minimum(numpy.maximum(z1, center_lens[m + bin_size] - range_lens / 2) + range_lens, z2) - range_lens / 3, y=9.0, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + bin_size + 1))
        
        if m == 0:
            plot[m, 1].set_title(r'$\mathrm{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 1].set_xlabel(r'$z$')
    
    for m in range(bin_size):
        plot[m, 2].plot(z_grid, dir_average_source[m, :], color='darkblue', linewidth=2.0, linestyle='-')
        
        plot[m, 2].plot(z_grid, stack_average_source[m, :], color='darkgreen', linewidth=2.0, linestyle='-')
        
        plot[m, 2].plot(z_grid, product_average_source[m, :], color='darkorange', linewidth=2.0, linestyle='-')
        
        plot[m, 2].plot(z_grid, truth_average_source[m, :], color='black', linewidth=2.0, linestyle='-')
        
        plot[m, 2].fill_betweenx(y=[0, 6], x1=bin_source[m], x2=bin_source[m + 1], color='gray', alpha=0.5)
        
        plot[m, 2].set_ylim(0, 6)
        plot[m, 2].set_xlim(numpy.maximum(z1, center_source[m] - range_source[m] / 2), numpy.minimum(numpy.maximum(z1, center_source[m] - range_source[m] / 2) + range_source[m], z2))
        
        plot[m, 2].set_yticks([2, 4, 6])
        plot[m, 2].text(x=numpy.minimum(numpy.maximum(z1, center_source[m] - range_source[m] / 2) + range_source[m], z2) - range_source[m] / 3, y=4.5, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1))
        
        if m == 0:
            plot[m, 2].set_title(r'$\mathrm{Source}$')
        
        if m == bin_size - 1:
            plot[m, 2].set_xlabel(r'$z$')
    
    figure.subplots_adjust(wspace=0.2, hspace=0.2)
    figure.savefig(os.path.join(analyze_folder, '{}/{}/MARGINAL/FIGURE.pdf'.format(tag, name)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
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
    PARSE.add_argument('--name', type=str, required=True, help='The name of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, FOLDER)