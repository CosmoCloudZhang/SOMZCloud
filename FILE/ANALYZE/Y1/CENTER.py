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
    ensemble_folder = os.path.join(folder, 'ENSEMBLE/')
    analysis_folder = os.path.join(folder, 'ANALYSIS/')
    
    # Summarize
    with h5py.File(os.path.join(ensemble_folder, '{}/SOM.hdf5'.format(tag)), 'r') as file:
        som_lens = file['lens']['ensemble'][...]
        som_source = file['source']['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/MODEL.hdf5'.format(tag)), 'r') as file:
        model_lens = file['lens']['ensemble'][...]
        model_source = file['source']['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/PRODUCT.hdf5'.format(tag)), 'r') as file:
        product_lens = file['lens']['ensemble'][...]
        product_source = file['source']['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/FIDUCIAL.hdf5'.format(tag)), 'r') as file:
        fiducial_lens = file['lens']['ensemble'][...]
        fiducial_source = file['source']['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/HISTOGRAM.hdf5'.format(tag)), 'r') as file:
        histogram_lens = file['lens']['ensemble'][...]
        histogram_source = file['source']['ensemble'][...]
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Center
    som_center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * som_lens, axis=2)
    som_center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * som_source, axis=2)
    
    model_center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * model_lens, axis=2)
    model_center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * model_source, axis=2)
    
    product_center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * product_lens, axis=2)
    product_center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * product_source, axis=2)
    
    fiducial_center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * fiducial_lens, axis=2)
    fiducial_center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * fiducial_source, axis=2)
    
    histogram_center_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * histogram_lens, axis=2)
    histogram_center_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * histogram_source, axis=2)
    
    # Average
    som_average_lens = numpy.mean(som_center_lens, axis=0)
    som_average_source = numpy.mean(som_center_source, axis=0)
    
    model_average_lens = numpy.mean(model_center_lens, axis=0)
    model_average_source = numpy.mean(model_center_source, axis=0)
    
    product_average_lens = numpy.mean(product_center_lens, axis=0)
    product_average_source = numpy.mean(product_center_source, axis=0)
    
    histogram_average_lens = numpy.mean(histogram_center_lens, axis=0)
    histogram_average_source = numpy.mean(histogram_center_source, axis=0)
    
    # Delta
    som_delta_lens = numpy.abs(som_average_lens - histogram_average_lens) / (1 + histogram_average_lens)
    som_delta_source = numpy.abs(som_average_source - histogram_average_source) / (1 + histogram_average_source)
    
    model_delta_lens = numpy.abs(model_average_lens - histogram_average_lens) / (1 + histogram_average_lens)
    model_delta_source = numpy.abs(model_average_source - histogram_average_source) / (1 + histogram_average_source)
    
    product_delta_lens = numpy.abs(product_average_lens - histogram_average_lens) / (1 + histogram_average_lens)
    product_delta_source = numpy.abs(product_average_source - histogram_average_source) / (1 + histogram_average_source)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot
    size = 100
    bin_size = 5
    lens_range = 0.03 * (1 + histogram_average_lens)
    source_range = 0.06 * (1 + histogram_average_source)
    figure, plot = pyplot.subplots(nrows=bin_size, ncols=2, figsize=(12, 3 * bin_size))
    
    for m in range(bin_size):
        
        plot[m, 0].hist(som_center_lens[:, m], bins=size, range=(histogram_average_lens[m] - lens_range[m], histogram_average_lens[m] + lens_range[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(model_center_lens[:, m], bins=size, range=(histogram_average_lens[m] - lens_range[m], histogram_average_lens[m] + lens_range[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(product_center_lens[:, m], bins=size, range=(histogram_average_lens[m] - lens_range[m], histogram_average_lens[m] + lens_range[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(fiducial_center_lens[:, m], bins=size, range=(histogram_average_lens[m] - lens_range[m], histogram_average_lens[m] + lens_range[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(histogram_center_lens[:, m], bins=size, range=(histogram_average_lens[m] - lens_range[m], histogram_average_lens[m] + lens_range[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].text(x=histogram_average_lens[m] + lens_range[m] * 0.6, y=200, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1), fontsize=20)
        
        plot[m, 0].text(x=histogram_average_lens[m] - lens_range[m] * 0.9, y=200, s=r'$\delta^\mathrm{SOM}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(som_delta_lens[m]), fontsize=20, color='darkblue')
        
        plot[m, 0].text(x=histogram_average_lens[m] - lens_range[m] * 0.9, y=100, s=r'$\delta^\mathrm{Model}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(model_delta_lens[m]), fontsize=20, color='darkgreen')
        
        plot[m, 0].text(x=histogram_average_lens[m] + lens_range[m] * 0.3, y=100, s=r'$\delta^\mathrm{Product}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(product_delta_lens[m]), fontsize=20, color='darkorange')
        
        plot[m, 0].set_ylim(10, 400)
        plot[m, 0].set_xlim(histogram_average_lens[m] - lens_range[m], histogram_average_lens[m] + lens_range[m])
        
        plot[m, 0].set_yscale('log')
        plot[m, 0].set_xlabel(r'$\langle z \rangle$')
        plot[m, 0].set_ylabel(r'$\phi \left( \langle z \rangle \right)$')
        
        if m == 0:
            plot[m, 0].set_title(r'$\mathrm{Lens}$')
    
    for m in range(bin_size):
        plot[m, 1].hist(som_center_source[:, m], bins=size, range=(histogram_average_source[m] - source_range[m], histogram_average_source[m] + source_range[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(model_center_source[:, m], bins=size, range=(histogram_average_source[m] - source_range[m], histogram_average_source[m] + source_range[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(product_center_source[:, m], bins=size, range=(histogram_average_source[m] - source_range[m], histogram_average_source[m] + source_range[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(fiducial_center_source[:, m], bins=size, range=(histogram_average_source[m] - source_range[m], histogram_average_source[m] + source_range[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(histogram_center_source[:, m], bins=size, range=(histogram_average_source[m] - source_range[m], histogram_average_source[m] + source_range[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].text(x=histogram_average_source[m] + source_range[m] * 0.6, y=200, s=r'$\mathrm{Bin} \, ' + r'{}$'.format(m + 1), fontsize=20)
        
        plot[m, 1].text(x=histogram_average_source[m] - source_range[m] * 0.9, y=200, s=r'$\delta^\mathrm{SOM}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(som_delta_source[m]), fontsize=20, color='darkblue')
        
        plot[m, 1].text(x=histogram_average_source[m] - source_range[m] * 0.9, y=100, s=r'$\delta^\mathrm{Model}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(model_delta_source[m]), fontsize=20, color='darkgreen')
        
        plot[m, 1].text(x=histogram_average_source[m] + source_range[m] * 0.3, y=100, s=r'$\delta^\mathrm{Product}_{\bar{\langle z \rangle}} = ' + r'{:.3f}$'.format(product_delta_source[m]), fontsize=20, color='darkorange')
        
        plot[m, 1].set_ylim(10, 400)
        plot[m, 1].set_xlim(histogram_average_source[m] - source_range[m], histogram_average_source[m] + source_range[m])
        
        plot[m, 1].set_yscale('log')
        plot[m, 1].set_xlabel(r'$\langle z \rangle$')
        plot[m, 1].set_ylabel(r'$\phi \left( \langle z \rangle \right)$')
        
        if m == 0:
            plot[m, 1].set_title(r'$\mathrm{Source}$')
    
    os.makedirs(analysis_folder, exist_ok=True)
    os.makedirs(os.path.join(analysis_folder, '{}/CENTER/'.format(tag)), exist_ok=True)
    
    figure.subplots_adjust(wspace=0.2, hspace=0.2)
    figure.savefig(os.path.join(analysis_folder, '{}/CENTER/FIGURE.pdf'.format(tag)), format='pdf', bbox_inches='tight')
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Analysis Center')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)