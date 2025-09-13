import os
import time
import h5py
import numpy
import scipy
import argparse
from matplotlib import pyplot


def main(tag, name, index, folder):
    '''
    Plot the average conditional ensemble redshift distribution
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
        index (int): The index of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Name: {} Index: {}'.format(name, index))
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    os.makedirs(os.path.join(analyze_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(analyze_folder, '{}/{}/CONDITIONAL/'.format(tag, name)), exist_ok=True)
    
    # Bin
    with h5py.File(os.path.join(model_folder, '{}/TARGET/DATA0.hdf5'.format(tag)), 'r') as file:
        bin_lens = file['bin_lens'][...]
        bin_source = file['bin_source'][...]
    
    # Summarize Lens
    with h5py.File(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}/DIR.hdf5'.format(tag, name, index)), 'r') as file:
        dir_lens = file['average'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}/STACK.hdf5'.format(tag, name, index)), 'r') as file:
        stack_lens = file['average'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}/PRODUCT.hdf5'.format(tag, name, index)), 'r') as file:
        product_lens = file['average'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}/TRUTH.hdf5'.format(tag, name, index)), 'r') as file:
        truth_lens = file['average'][...]
    
    # Summarize Source
    with h5py.File(os.path.join(summarize_folder, '{}/{}/SOURCE/SOURCE{}/DIR.hdf5'.format(tag, name, index)), 'r') as file:
        dir_source = file['average'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/{}/SOURCE/SOURCE{}/STACK.hdf5'.format(tag, name, index)), 'r') as file:
        stack_source = file['average'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/{}/SOURCE/SOURCE{}/PRODUCT.hdf5'.format(tag, name, index)), 'r') as file:
        product_source = file['average'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/{}/SOURCE/SOURCE{}/TRUTH.hdf5'.format(tag, name, index)), 'r') as file:
        truth_source = file['average'][...]
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * truth_lens, axis=1)
    center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * truth_source, axis=1)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 30
    
    # Plot
    bin_size = 5
    range_lens = 0.6
    range_source = 1.8
    figure, plot = pyplot.subplots(nrows=bin_size, ncols=3, figsize=(18, 5 * bin_size))
    
    for m in range(bin_size):
        plot[m, 0].plot(z_grid, dir_lens[m, :], color='darkblue', linewidth=1.5, linestyle='-')
        
        plot[m, 0].plot(z_grid, stack_lens[m, :], color='darkgreen', linewidth=1.5, linestyle='-')
        
        plot[m, 0].plot(z_grid, product_lens[m, :], color='darkorange', linewidth=1.5, linestyle='-')
        
        plot[m, 0].plot(z_grid, truth_lens[m, :], color='black', linewidth=1.5, linestyle='-')
        
        plot[m, 0].fill_betweenx(y=[0, 12], x1=bin_lens[m], x2=bin_lens[m + 1], color='gray', alpha=0.5)
        
        plot[m, 0].set_ylim(0, 12)
        plot[m, 0].set_xlim(numpy.maximum(z1, center_lens[m] - range_lens / 2), numpy.minimum(numpy.maximum(z1, center_lens[m] - range_lens / 2) + range_lens, z2))
        
        plot[m, 0].set_yticks([3, 6, 9, 12])
        plot[m, 0].set_ylabel(r'$\phi \left( z | \mathcal{S}_n, \mathcal{P}_n \right)$')
        plot[m, 0].text(x=numpy.minimum(numpy.maximum(z1, center_lens[m] - range_lens / 2) + range_lens, z2) - range_lens / 3, y=9.0, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1))
        
        if m == 0:
            plot[m, 0].set_title(r'$\mathtt{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 0].set_xlabel(r'$z$')
    
    for m in range(bin_size):
        plot[m, 1].plot(z_grid, dir_lens[m + bin_size, :], color='darkblue', linewidth=1.5, linestyle='-')
        
        plot[m, 1].plot(z_grid, stack_lens[m + bin_size, :], color='darkgreen', linewidth=1.5, linestyle='-')
        
        plot[m, 1].plot(z_grid, product_lens[m + bin_size, :], color='darkorange', linewidth=1.5, linestyle='-')
        
        plot[m, 1].plot(z_grid, truth_lens[m + bin_size, :], color='black', linewidth=1.5, linestyle='-')
        
        plot[m, 1].fill_betweenx(y=[0, 12], x1=bin_lens[m + bin_size], x2=bin_lens[m + bin_size + 1], color='gray', alpha=0.5)
        
        plot[m, 1].set_ylim(0, 12)
        plot[m, 1].set_xlim(numpy.maximum(z1, center_lens[m + bin_size] - range_lens / 2), numpy.minimum(numpy.maximum(z1, center_lens[m + bin_size] - range_lens / 2) + range_lens, z2))
        
        plot[m, 1].set_yticks([3, 6, 9, 12])
        plot[m, 1].text(x=numpy.minimum(numpy.maximum(z1, center_lens[m + bin_size] - range_lens / 2) + range_lens, z2) - range_lens / 3, y=9.0, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + bin_size + 1))
        
        if m == 0:
            plot[m, 1].set_title(r'$\mathtt{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 1].set_xlabel(r'$z$')
    
    for m in range(bin_size):
        plot[m, 2].plot(z_grid, dir_source[m, :], color='darkblue', linewidth=1.5, linestyle='-')
        
        plot[m, 2].plot(z_grid, stack_source[m, :], color='darkgreen', linewidth=1.5, linestyle='-')
        
        plot[m, 2].plot(z_grid, product_source[m, :], color='darkorange', linewidth=1.5, linestyle='-')
        
        plot[m, 2].plot(z_grid, truth_source[m, :], color='black', linewidth=1.5, linestyle='-')
        
        plot[m, 2].fill_betweenx(y=[0, 6], x1=bin_source[m], x2=bin_source[m + 1], color='gray', alpha=0.5)
        
        plot[m, 2].set_ylim(0, 6)
        plot[m, 2].set_xlim(numpy.maximum(z1, center_source[m] - range_source / 2), numpy.minimum(numpy.maximum(z1, center_source[m] - range_source / 2) + range_source, z2))
        
        plot[m, 2].set_yticks([2, 4, 6])
        plot[m, 2].text(x=numpy.minimum(numpy.maximum(z1, center_source[m] - range_source / 2) + range_source, z2) - range_source / 3, y=4.5, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1))
        
        if m == 0:
            plot[m, 2].set_title(r'$\mathtt{Source}$')
        
        if m == bin_size - 1:
            plot[m, 2].set_xlabel(r'$z$')
        
    figure.subplots_adjust(wspace=0.2, hspace=0.2)
    figure.savefig(os.path.join(analyze_folder, '{}/{}/CONDITIONAL/FIGURE{}.pdf'.format(tag, name, index)), format='pdf', bbox_inches='tight')
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Condition')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, INDEX, FOLDER)