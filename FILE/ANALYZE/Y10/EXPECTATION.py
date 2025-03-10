import os
import h5py
import time
import numpy
import scipy
import argparse
from matplotlib import pyplot


def main(tag, folder):
    '''
    Plot the center of the lens and source redshift distributions
    
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
    
    label_list = ['ZERO', 'HALF', 'UNITY', 'DOUBLE']
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
        
        # Delta
        som_delta_lens = numpy.abs(som_center_lens - histogram_center_lens) / (1 + histogram_center_lens)
        som_delta_source = numpy.abs(som_center_source - histogram_center_source) / (1 + histogram_center_source)
        
        model_delta_lens = numpy.abs(model_center_lens - histogram_center_lens) / (1 + histogram_center_lens)
        model_delta_source = numpy.abs(model_center_source - histogram_center_source) / (1 + histogram_center_source)
        
        product_delta_lens = numpy.abs(product_center_lens - histogram_center_lens) / (1 + histogram_center_lens)
        product_delta_source = numpy.abs(product_center_source - histogram_center_source) / (1 + histogram_center_source)
        
        fiducial_delta_lens = numpy.abs(fiducial_center_lens - histogram_center_lens) / (1 + histogram_center_lens)
        fiducial_delta_source = numpy.abs(fiducial_center_source - histogram_center_source) / (1 + histogram_center_source)
        
        # Configuration
        os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
        pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
        pyplot.rcParams['text.usetex'] = True
        pyplot.rcParams['font.size'] = 20
        
        # Plot
        size = 100
        bin_size = 5
        lens_range = 0.02 * (1 + histogram_center_lens)
        source_range = 0.04 * (1 + histogram_center_source)
        figure, plot = pyplot.subplots(nrows=bin_size, ncols=3, figsize=(18, 3 * bin_size))
        
        for m in range(bin_size):
            
            plot[m, 0].hist(som_expectation_lens[:, m], bins=size, range=(histogram_center_lens[m] - lens_range[m], histogram_center_lens[m] + lens_range[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].hist(model_expectation_lens[:, m], bins=size, range=(histogram_center_lens[m] - lens_range[m], histogram_center_lens[m] + lens_range[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].hist(product_expectation_lens[:, m], bins=size, range=(histogram_center_lens[m] - lens_range[m], histogram_center_lens[m] + lens_range[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].hist(fiducial_expectation_lens[:, m], bins=size, range=(histogram_center_lens[m] - lens_range[m], histogram_center_lens[m] + lens_range[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].hist(histogram_expectation_lens[:, m], bins=size, range=(histogram_center_lens[m] - lens_range[m], histogram_center_lens[m] + lens_range[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].text(x=histogram_center_lens[m] - lens_range[m] * 0.9, y=250, s=r'$\delta^\mathrm{SOM}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(som_delta_lens[m]), fontsize=15, color='darkblue')
            
            plot[m, 0].text(x=histogram_center_lens[m] - lens_range[m] * 0.9, y=100, s=r'$\delta^\mathrm{Model}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(model_delta_lens[m]), fontsize=15, color='darkgreen')
            
            plot[m, 0].text(x=histogram_center_lens[m] + lens_range[m] * 0.1, y=250, s=r'$\delta^\mathrm{Product}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(product_delta_lens[m]), fontsize=15, color='darkorange')
            
            plot[m, 0].text(x=histogram_center_lens[m] + lens_range[m] * 0.1, y=100, s=r'$\delta^\mathrm{Fiducial}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(fiducial_delta_lens[m]), fontsize=15, color='darkred')
            
            plot[m, 0].set_ylim(10, 500)
            plot[m, 0].set_xlim(histogram_center_lens[m] - lens_range[m], histogram_center_lens[m] + lens_range[m])
            
            plot[m, 0].set_yscale('log')
            plot[m, 0].set_xlabel(r'$\langle z \rangle$')
            plot[m, 0].set_ylabel(r'$\phi \left( \langle z \rangle \right)$')
            
            if m == 0:
                plot[m, 0].set_title(r'$\mathrm{Lens}$')
        
        for m in range(bin_size):
            
            plot[m, 0].hist(som_expectation_lens[:, m + bin_size], bins=size, range=(histogram_center_lens[m + bin_size] - lens_range[m], histogram_center_lens[m + bin_size] + lens_range[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].hist(model_expectation_lens[:, m + bin_size], bins=size, range=(histogram_center_lens[m + bin_size] - lens_range[m], histogram_center_lens[m + bin_size] + lens_range[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].hist(product_expectation_lens[:, m + bin_size], bins=size, range=(histogram_center_lens[m + bin_size] - lens_range[m], histogram_center_lens[m + bin_size] + lens_range[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].hist(fiducial_expectation_lens[:, m + bin_size], bins=size, range=(histogram_center_lens[m + bin_size] - lens_range[m], histogram_center_lens[m + bin_size] + lens_range[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].hist(histogram_expectation_lens[:, m + bin_size], bins=size, range=(histogram_center_lens[m + bin_size] - lens_range[m], histogram_center_lens[m + bin_size] + lens_range[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 0].text(x=histogram_center_lens[m + bin_size] - lens_range[m] * 0.9, y=250, s=r'$\delta^\mathrm{SOM}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(som_delta_lens[m + bin_size]), fontsize=15, color='darkblue')
            
            plot[m, 0].text(x=histogram_center_lens[m + bin_size] - lens_range[m] * 0.9, y=100, s=r'$\delta^\mathrm{Model}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(model_delta_lens[m + bin_size]), fontsize=15, color='darkgreen')
            
            plot[m, 0].text(x=histogram_center_lens[m + bin_size] + lens_range[m] * 0.1, y=250, s=r'$\delta^\mathrm{Product}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(product_delta_lens[m + bin_size]), fontsize=15, color='darkorange')
            
            plot[m, 0].text(x=histogram_center_lens[m + bin_size] + lens_range[m] * 0.1, y=100, s=r'$\delta^\mathrm{Fiducial}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(fiducial_delta_lens[m + bin_size]), fontsize=15, color='darkred')
            
            plot[m, 0].set_ylim(10, 500)
            plot[m, 0].set_xlim(histogram_center_lens[m + bin_size] - lens_range[m], histogram_center_lens[m + bin_size] + lens_range[m])
            
            plot[m, 0].set_yscale('log')
            plot[m, 0].set_xlabel(r'$\langle z \rangle$')
            plot[m, 0].set_ylabel(r'$\phi \left( \langle z \rangle \right)$')
            
            if m == 0:
                plot[m, 0].set_title(r'$\mathrm{Lens}$')
        
        for m in range(bin_size):
            plot[m, 2].hist(som_expectation_source[:, m], bins=size, range=(histogram_center_source[m] - source_range[m], histogram_center_source[m] + source_range[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 2].hist(model_expectation_source[:, m], bins=size, range=(histogram_center_source[m] - source_range[m], histogram_center_source[m] + source_range[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 2].hist(product_expectation_source[:, m], bins=size, range=(histogram_center_source[m] - source_range[m], histogram_center_source[m] + source_range[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 2].hist(fiducial_expectation_source[:, m], bins=size, range=(histogram_center_source[m] - source_range[m], histogram_center_source[m] + source_range[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 2].hist(histogram_expectation_source[:, m], bins=size, range=(histogram_center_source[m] - source_range[m], histogram_center_source[m] + source_range[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
            
            plot[m, 2].text(x=histogram_center_source[m] - source_range[m] * 0.9, y=250, s=r'$\delta^\mathrm{SOM}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(som_delta_source[m]), fontsize=15, color='darkblue')
            
            plot[m, 2].text(x=histogram_center_source[m] - source_range[m] * 0.9, y=100, s=r'$\delta^\mathrm{Model}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(model_delta_source[m]), fontsize=15, color='darkgreen')
            
            plot[m, 2].text(x=histogram_center_source[m] + source_range[m] * 0.1, y=250, s=r'$\delta^\mathrm{Product}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(product_delta_source[m]), fontsize=15, color='darkorange')
            
            plot[m, 2].text(x=histogram_center_source[m] + source_range[m] * 0.1, y=100, s=r'$\delta^\mathrm{Fiducial}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(fiducial_delta_source[m]), fontsize=15, color='darkred')
            
            plot[m, 2].set_ylim(10, 500)
            plot[m, 2].set_xlim(histogram_center_source[m] - source_range[m], histogram_center_source[m] + source_range[m])
            
            plot[m, 2].set_yscale('log')
            plot[m, 2].set_xlabel(r'$\langle z \rangle$')
            plot[m, 2].set_ylabel(r'$\phi \left( \langle z \rangle \right)$')
            
            if m == 0:
                plot[m, 2].set_title(r'$\mathrm{Source}$')
        
        os.makedirs(analyze_folder, exist_ok=True)
        os.makedirs(os.path.join(analyze_folder, '{}/EXPECTATION/'.format(tag)), exist_ok=True)
        
        figure.subplots_adjust(wspace=0.2, hspace=0.2)
        figure.savefig(os.path.join(analyze_folder, '{}/EXPECTATION/FIGURE_{}.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
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