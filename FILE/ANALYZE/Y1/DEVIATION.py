import os
import h5py
import time
import numpy
import scipy
import argparse
from matplotlib import pyplot


def main(tag, folder):
    '''
    Plot the standard deviation of the lens and source redshift distributions
    
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
    
    label_list = ['ZERO', 'HALF', 'UNITY']
    for label in label_list:
        # Summarize
        with h5py.File(os.path.join(synthesize_folder, '{}/SOM_{}.hdf5'.format(tag, label)), 'r') as file:
            som_data_lens = file['lens']['data'][...]
            som_data_source = file['source']['data'][...]
            
            som_average_lens = file['lens']['average'][...]
            som_average_source = file['source']['average'][...]
        
        with h5py.File(os.path.join(synthesize_folder, '{}/MODEL_{}.hdf5'.format(tag, label)), 'r') as file:
            model_data_lens = file['lens']['data'][...]
            model_data_source = file['source']['data'][...]
            
            model_average_lens = file['lens']['average'][...]
            model_average_source = file['source']['average'][...]
        
        with h5py.File(os.path.join(synthesize_folder, '{}/PRODUCT_{}.hdf5'.format(tag, label)), 'r') as file:
            product_data_lens = file['lens']['data'][...]
            product_data_source = file['source']['data'][...]
            
            product_average_lens = file['lens']['average'][...]
            product_average_source = file['source']['average'][...]
        
        with h5py.File(os.path.join(synthesize_folder, '{}/FIDUCIAL_{}.hdf5'.format(tag, label)), 'r') as file:
            fiducial_data_lens = file['lens']['data'][...]
            fiducial_data_source = file['source']['data'][...]
            
            fiducial_average_lens = file['lens']['average'][...]
            fiducial_average_source = file['source']['average'][...]
        
        with h5py.File(os.path.join(synthesize_folder, '{}/HISTOGRAM_{}.hdf5'.format(tag, label)), 'r') as file:
            histogram_data_lens = file['lens']['data'][...]
            histogram_data_source = file['source']['data'][...]
            
            histogram_average_lens = file['lens']['average'][...]
            histogram_average_source = file['source']['average'][...]
        
        # Redshift
        z1 = 0.0
        z2 = 3.0
        grid_size = 300
        z_grid = numpy.linspace(z1, z2, grid_size + 1)
        
        # Expectation
        som_expectation_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * som_data_lens, axis=2)
        som_expectation_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * som_data_source, axis=2)
        
        model_expectation_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * model_data_lens, axis=2)
        model_expectation_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * model_data_source, axis=2)
        
        product_expectation_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * product_data_lens, axis=2)
        product_expectation_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * product_data_source, axis=2)
        
        fiducial_expectation_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * fiducial_data_lens, axis=2)
        fiducial_expectation_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * fiducial_data_source, axis=2)
        
        histogram_expectation_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * histogram_data_lens, axis=2)
        histogram_expectation_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * histogram_data_source, axis=2)
        
        # Center
        som_center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * som_average_lens, axis=1)
        som_center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * som_average_source, axis=1)
        
        model_center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * model_average_lens, axis=1)
        model_center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * model_average_source, axis=1)
        
        product_center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * product_average_lens, axis=1)
        product_center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * product_average_source, axis=1)
        
        fiducial_center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * fiducial_average_lens, axis=1)
        fiducial_center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * fiducial_average_source, axis=1)
        
        histogram_center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * histogram_average_lens, axis=1)
        histogram_center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * histogram_average_source, axis=1)
        
        # Deviation
        som_deviation_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, numpy.newaxis, :] * som_data_lens, axis=2) - numpy.square(som_expectation_lens))
        som_deviation_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, numpy.newaxis, :] * som_data_source, axis=2) - numpy.square(som_expectation_source))
        
        model_deviation_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, numpy.newaxis, :] * model_data_lens, axis=2) - numpy.square(model_expectation_lens))
        model_deviation_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, numpy.newaxis, :] * model_data_source, axis=2) - numpy.square(model_expectation_source))
        
        product_deviation_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, numpy.newaxis, :] * product_data_lens, axis=2) - numpy.square(product_expectation_lens))
        product_deviation_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, numpy.newaxis, :] * product_data_source, axis=2) - numpy.square(product_expectation_source))
        
        fiducial_deviation_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, numpy.newaxis, :] * fiducial_data_lens, axis=2) - numpy.square(fiducial_expectation_lens))
        fiducial_deviation_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, numpy.newaxis, :] * fiducial_data_source, axis=2) - numpy.square(fiducial_expectation_source))
        
        histogram_deviation_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, numpy.newaxis, :] * histogram_data_lens, axis=2) - numpy.square(histogram_expectation_lens))
        histogram_deviation_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, numpy.newaxis, :] * histogram_data_source, axis=2) - numpy.square(histogram_expectation_source))
        
        # Scatter
        som_scatter_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, :] * numpy.square(som_average_lens), axis=1) - numpy.square(som_center_lens))
        som_scatter_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, :] * numpy.square(som_average_source), axis=1) - numpy.square(som_center_source))
        
        model_scatter_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, :] * numpy.square(model_average_lens), axis=1) - numpy.square(model_center_lens))
        model_scatter_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, :] * numpy.square(model_average_source), axis=1) - numpy.square(model_center_source))
        
        product_scatter_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, :] * numpy.square(product_average_lens), axis=1) - numpy.square(product_center_lens))
        product_scatter_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, :] * numpy.square(product_average_source), axis=1) - numpy.square(product_center_source))
        
        fiducial_scatter_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, :] * numpy.square(fiducial_average_lens), axis=1) - numpy.square(fiducial_center_lens))
        fiducial_scatter_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, :] * numpy.square(fiducial_average_source), axis=1) - numpy.square(fiducial_center_source))
        
        histogram_scatter_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, :] * numpy.square(histogram_average_lens), axis=1) - numpy.square(histogram_center_lens))
        histogram_scatter_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid)[numpy.newaxis, :] * numpy.square(histogram_average_source), axis=1) - numpy.square(histogram_center_source))
        
        # Delta
        som_delta_lens = numpy.abs(som_scatter_lens - histogram_scatter_lens) / histogram_scatter_lens
        som_delta_source = numpy.abs(som_scatter_source - histogram_scatter_source) / histogram_scatter_source
        
        model_delta_lens = numpy.abs(model_scatter_lens - histogram_scatter_lens) / histogram_scatter_lens
        model_delta_source = numpy.abs(model_scatter_source - histogram_scatter_source) / histogram_scatter_source
        
        product_delta_lens = numpy.abs(product_scatter_lens - histogram_scatter_lens) / histogram_scatter_lens
        product_delta_source = numpy.abs(product_scatter_source - histogram_scatter_source) / histogram_scatter_source
        
        fiducial_delta_lens = numpy.abs(fiducial_scatter_lens - histogram_scatter_lens) / histogram_scatter_lens
        fiducial_delta_source = numpy.abs(fiducial_scatter_source - histogram_scatter_source) / histogram_scatter_source
        
        # Configuration
        os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
        pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
        pyplot.rcParams['text.usetex'] = True
        pyplot.rcParams['font.size'] = 20
        
        # Plot
        size = 100
        bin_size = 5
        lens_range = 0.25 * histogram_scatter_lens
        source_range = 0.50 * histogram_scatter_source
        figure, plot = pyplot.subplots(nrows=bin_size, ncols=2, figsize=(12, 3 * bin_size))
        
        for m in range(bin_size):
            
            plot[m, 0].hist(som_deviation_lens[:, m], bins=size, range=(histogram_scatter_lens[m] - lens_range[m], histogram_scatter_lens[m] + lens_range[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].hist(model_deviation_lens[:, m], bins=size, range=(histogram_scatter_lens[m] - lens_range[m], histogram_scatter_lens[m] + lens_range[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].hist(product_deviation_lens[:, m], bins=size, range=(histogram_scatter_lens[m] - lens_range[m], histogram_scatter_lens[m] + lens_range[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].hist(fiducial_deviation_lens[:, m], bins=size, range=(histogram_scatter_lens[m] - lens_range[m], histogram_scatter_lens[m] + lens_range[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].hist(histogram_deviation_lens[:, m], bins=size, range=(histogram_scatter_lens[m] - lens_range[m], histogram_scatter_lens[m] + lens_range[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].text(x=histogram_scatter_lens[m] - lens_range[m] * 0.75, y=250, s=r'$\delta^\mathrm{SOM}_{\bar{\langle \varsigma \rangle}} = ' + r'{:.3f}$'.format(som_delta_lens[m]), fontsize=15, color='darkblue')
            
            plot[m, 0].text(x=histogram_scatter_lens[m] - lens_range[m] * 0.75, y=100, s=r'$\delta^\mathrm{Model}_{\bar{\langle \varsigma \rangle}} = ' + r'{:.3f}$'.format(model_delta_lens[m]), fontsize=15, color='darkgreen')
            
            plot[m, 0].text(x=histogram_scatter_lens[m] + lens_range[m] * 0.25, y=250, s=r'$\delta^\mathrm{Product}_{\bar{\langle \varsigma \rangle}} = ' + r'{:.3f}$'.format(product_delta_lens[m]), fontsize=15, color='darkorange')
            
            plot[m, 0].text(x=histogram_scatter_lens[m] + lens_range[m] * 0.25, y=100, s=r'$\delta^\mathrm{Fiducial}_{\bar{\langle \varsigma \rangle}} = ' + r'{:.3f}$'.format(fiducial_delta_lens[m]), fontsize=15, color='darkred')
            
            plot[m, 0].set_ylim(10, 500)
            plot[m, 0].set_xlim(histogram_scatter_lens[m] - lens_range[m], histogram_scatter_lens[m] + lens_range[m])
            
            plot[m, 0].set_yscale('log')
            plot[m, 0].set_xlabel(r'$\langle \varsigma \rangle$')
            
            if m == 0:
                plot[m, 0].set_title(r'$\mathrm{Lens}$')
        
        for m in range(bin_size):
            plot[m, 1].hist(som_deviation_source[:, m], bins=size, range=(histogram_scatter_source[m] - source_range[m], histogram_scatter_source[m] + source_range[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 1].hist(model_deviation_source[:, m], bins=size, range=(histogram_scatter_source[m] - source_range[m], histogram_scatter_source[m] + source_range[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 1].hist(product_deviation_source[:, m], bins=size, range=(histogram_scatter_source[m] - source_range[m], histogram_scatter_source[m] + source_range[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 1].hist(fiducial_deviation_source[:, m], bins=size, range=(histogram_scatter_source[m] - source_range[m], histogram_scatter_source[m] + source_range[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 1].hist(histogram_deviation_source[:, m], bins=size, range=(histogram_scatter_source[m] - source_range[m], histogram_scatter_source[m] + source_range[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 1].text(x=histogram_scatter_source[m] - source_range[m] * 0.75, y=250, s=r'$\delta^\mathrm{SOM}_{\bar{\langle \varsigma \rangle}} = ' + r'{:.3f}$'.format(som_delta_source[m]), fontsize=15, color='darkblue')
            
            plot[m, 1].text(x=histogram_scatter_source[m] - source_range[m] * 0.75, y=100, s=r'$\delta^\mathrm{Model}_{\bar{\langle \varsigma \rangle}} = ' + r'{:.3f}$'.format(model_delta_source[m]), fontsize=15, color='darkgreen')
            
            plot[m, 1].text(x=histogram_scatter_source[m] + source_range[m] * 0.25, y=250, s=r'$\delta^\mathrm{Product}_{\bar{\langle \varsigma \rangle}} = ' + r'{:.3f}$'.format(product_delta_source[m]), fontsize=15, color='darkorange')
            
            plot[m, 1].text(x=histogram_scatter_source[m] + source_range[m] * 0.25, y=100, s=r'$\delta^\mathrm{Fiducial}_{\bar{\langle \varsigma \rangle}} = ' + r'{:.3f}$'.format(fiducial_delta_source[m]), fontsize=15, color='darkred')
            
            plot[m, 1].set_ylim(10, 500)
            plot[m, 1].set_xlim(histogram_scatter_source[m] - source_range[m], histogram_scatter_source[m] + source_range[m])
            
            plot[m, 1].set_yscale('log')
            plot[m, 1].set_xlabel(r'$\langle \varsigma \rangle$')
            
            if m == 0:
                plot[m, 1].set_title(r'$\mathrm{Source}$')
        
        os.makedirs(analyze_folder, exist_ok=True)
        os.makedirs(os.path.join(analyze_folder, '{}/DEVIATION/'.format(tag)), exist_ok=True)
        
        figure.subplots_adjust(wspace=0.2, hspace=0.2)
        figure.savefig(os.path.join(analyze_folder, '{}/DEVIATION/FIGURE_{}.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Analyze Expectation')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)