import os
import time
import h5py
import numpy
import scipy
import argparse
from matplotlib import pyplot


def main(tag, number, folder):
    '''
    Plot the bias of conditional ensemble redshift distribution
    
    Arguments:
        tag (str): The tag of the configuration
        number (int): The number of the datasets
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    figure_folder = os.path.join(folder, 'FIGURE/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Bin
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA0.hdf5'.format(tag)), 'r') as file:
        bin_lens_size = len(file['bin_lens'][...]) - 1
    
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA0.hdf5'.format(tag)), 'r') as file:
        bin_source_size = len(file['bin_source'][...]) - 1
    
    # Delta
    som_delta_lens = numpy.zeros((bin_lens_size, number))
    som_delta_source = numpy.zeros((bin_source_size, number))
    
    model_delta_lens = numpy.zeros((bin_lens_size, number))
    model_delta_source = numpy.zeros((bin_source_size, number))
    
    product_delta_lens = numpy.zeros((bin_lens_size, number))
    product_delta_source = numpy.zeros((bin_source_size, number))
    
    sigma_lens = numpy.zeros((bin_lens_size, number))
    sigma_source = numpy.zeros((bin_source_size, number))
    
    metric = numpy.zeros(number)
    metric_lens = numpy.zeros((bin_lens_size, number))
    metric_source = numpy.zeros((bin_source_size, number))
    
    for index in range(1, number + 1):
        print('Index: {}'.format(index))
        # Combination
        with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            metric[index - 1] = file['meta']['metric'][...]
        
        # Summarize Lens
        with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/SOM.hdf5'.format(tag, index)), 'r') as file:
            som_lens = file['average'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/MODEL.hdf5'.format(tag, index)), 'r') as file:
            model_lens = file['average'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/PRODUCT.hdf5'.format(tag, index)), 'r') as file:
            product_lens = file['average'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/HISTOGRAM.hdf5'.format(tag, index)), 'r') as file:
            histogram_lens = file['average'][...]
            sigma_lens[:, index - 1] = numpy.mean(file['sigma'][...], axis=1)
        
        metric_lens[:, index - 1] = numpy.sqrt(metric[index - 1] * sigma_lens[:, index - 1])
        histogram_mean_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * histogram_lens, axis=1)
        
        som_mean_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * som_lens, axis=1)
        som_delta_lens[:, index - 1] = numpy.abs(som_mean_lens - histogram_mean_lens) / (1 + histogram_mean_lens)
        
        model_mean_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * model_lens, axis=1)
        model_delta_lens[:, index - 1] = numpy.abs(model_mean_lens - histogram_mean_lens) / (1 + histogram_mean_lens)
        
        product_mean_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * product_lens, axis=1)
        product_delta_lens[:, index - 1] = numpy.abs(product_mean_lens - histogram_mean_lens) / (1 + histogram_mean_lens)
        
        # Summarize Source
        with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/SOM.hdf5'.format(tag, number)), 'r') as file:
            som_source = file['average'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/MODEL.hdf5'.format(tag, number)), 'r') as file:
            model_source = file['average'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/PRODUCT.hdf5'.format(tag, number)), 'r') as file:
            product_source = file['average'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/HISTOGRAM.hdf5'.format(tag, number)), 'r') as file:
            histogram_source = file['average'][...]
            sigma_source[:, index - 1] = numpy.mean(file['sigma'][...], axis=1)
        
        metric_source[:, index - 1] = numpy.sqrt(metric[index - 1] * sigma_source[:, index - 1])
        histogram_mean_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * histogram_source, axis=1)
        
        som_mean_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * som_source, axis=1)
        som_delta_source[:, index - 1] = numpy.abs(som_mean_source - histogram_mean_source) / (1 + histogram_mean_source)
        
        model_mean_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * model_source, axis=1)
        model_delta_source[:, index - 1] = numpy.abs(model_mean_source - histogram_mean_source) / (1 + histogram_mean_source)
        
        product_mean_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * product_source, axis=1)
        product_delta_source[:, index - 1] = numpy.abs(product_mean_source - histogram_mean_source) / (1 + histogram_mean_source)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot
    bin_size = 5
    factor_lens = 0.003
    factor_source = 0.001
    figure, plot = pyplot.subplots(nrows=bin_size, ncols=3, figsize=(18, 3 * bin_size))
    
    for m in range(bin_size):
        plot[m, 0].plot([0.0, 0.6], [factor_lens, factor_lens], color='black', linestyle='--')
        
        plot[m, 0].scatter(metric_lens[m, :], som_delta_lens[m, :], color='darkblue', marker='o', s=10)
        
        plot[m, 0].scatter(metric_lens[m, :], model_delta_lens[m, :], color='darkgreen', marker='o', s=10)
        
        plot[m, 0].scatter(metric_lens[m, :], product_delta_lens[m, :], color='darkorange', marker='o', s=10)
        
        plot[m, 0].set_xlim(0.0, 0.6)
        plot[m, 0].set_ylim(5e-5, 2e-1)
        
        plot[m, 0].set_yscale('log')
        plot[m, 0].set_ylabel(r'$\delta_{\langle z \rangle}$')
        plot[m, 0].text(x=0.50, y=0.08, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1), fontsize=20)
        
        if m == 0:
            plot[m, 0].set_title(r'$\mathrm{Lens}$')
        
        if m < bin_size - 1:
            plot[m, 0].set_xticklabels([])
        else:
            plot[m, 0].set_xticks([0.0, 0.2, 0.4, 0.6])
            plot[m, 0].set_xlabel(r'$\sigma_\mathrm{SOM}$')
    
    for m in range(bin_size):
        plot[m, 1].plot([0.0, 0.6], [factor_lens, factor_lens], color='black', linestyle='--')
        
        plot[m, 1].scatter(metric_lens[m + bin_size, :], som_delta_lens[m + bin_size, :], color='darkblue', marker='o', s=10)
        
        plot[m, 1].scatter(metric_lens[m + bin_size, :], model_delta_lens[m + bin_size, :], color='darkgreen', marker='o', s=10)
        
        plot[m, 1].scatter(metric_lens[m + bin_size, :], product_delta_lens[m + bin_size, :], color='darkorange', marker='o', s=10)
        
        plot[m, 1].set_xlim(0.0, 0.6)
        plot[m, 1].set_ylim(5e-5, 2e-1)
        
        plot[m, 1].set_yscale('log')
        plot[m, 1].set_yticklabels([])
        plot[m, 1].text(x=0.50, y=0.08, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1), fontsize=20)
        
        if m == 0:
            plot[m, 1].set_title(r'$\mathrm{Lens}$')
        
        if m < bin_size - 1:
            plot[m, 1].set_xticklabels([])
        else:
            plot[m, 1].set_xticks([0.2, 0.4, 0.6])
            plot[m, 1].set_xlabel(r'$\sigma_\mathrm{SOM}$')
    
    for m in range(bin_size):
        plot[m, 2].plot([0.0, 0.6], [factor_source, factor_source], color='black', linestyle='--')
        
        plot[m, 2].scatter(metric_source[m, :], som_delta_source[m, :], color='darkblue', marker='o', s=10)
        
        plot[m, 2].scatter(metric_source[m, :], model_delta_source[m, :], color='darkgreen', marker='o', s=10)
        
        plot[m, 2].scatter(metric_source[m, :], product_delta_source[m, :], color='darkorange', marker='o', s=10)
        
        plot[m, 2].set_xlim(0.0, 0.6)
        plot[m, 2].set_ylim(5e-5, 2e-1)
        
        plot[m, 2].set_yscale('log')
        plot[m, 2].set_yticklabels([])
        plot[m, 2].text(x=0.50, y=0.08, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1), fontsize=20)
        
        if m == 0:
            plot[m, 2].set_title(r'$\mathrm{Source}$')
        
        if m < bin_size - 1:
            plot[m, 2].set_xticklabels([])
        else:
            plot[m, 2].set_xticks([0.2, 0.4, 0.6])
            plot[m, 2].set_xlabel(r'$\sigma_\mathrm{SOM}$')
    
    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/DELTA/'.format(tag)), exist_ok=True)
    
    figure.subplots_adjust(wspace=0.0, hspace=0.0)
    figure.savefig(os.path.join(figure_folder, '{}/DELTA/FIGURE.png'.format(tag)), bbox_inches='tight', format='png', dpi=512)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Delta')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)