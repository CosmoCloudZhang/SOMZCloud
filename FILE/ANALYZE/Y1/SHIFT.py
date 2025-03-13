import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot
from matplotlib import colors


def plot_covariance(bin_size, map_size, covariance):
    '''
    Plot the shift covariance matrix
    
    Arguments:
        bin_size (int): The number of bins
        map_size (int): The size of the map
        covariance (numpy.ndarray): The covariance matrix
    
    Returns:
        figure (matplotlib.figure.Figure): The figure
    '''
    norm = colors.Normalize(vmin = -0.5, vmax = +0.5)
    figure, plot = pyplot.subplots(nrows = bin_size, ncols = bin_size, figsize = (3 * bin_size, 3 * bin_size))
    
    for m in range(bin_size):
        for n in range(bin_size):
            
            map_covariance = covariance[n * map_size: (n + 1) * map_size, m * map_size: (m + 1) * map_size]
            image = plot[n, m].imshow(map_covariance, norm = norm, cmap = 'seismic', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.0, hspace = 0.0)
    
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\mathcal{C} \: [\phi^{i} (z_1), \phi^{j} (z_2)]$')
    return figure


def main(tag, folder):
    '''
    Plot the shift covariance matrix
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    
    # Bin
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA0.hdf5'.format(tag)), 'r') as file:
        bin_lens = file['bin_lens'][...]
        bin_source = file['bin_source'][...]
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_delta = (z2 - z1) / grid_size
    
    # Index
    map_lens_size = 40
    map_source_size = 100
    
    index_lower_lens = numpy.maximum(0, numpy.array(numpy.round(bin_lens[:-1] / z_delta, decimals=0), dtype='int32') - map_lens_size // 4)
    index_upper_lens = index_lower_lens + map_lens_size
    
    index_lower_source = numpy.maximum(0, numpy.array(numpy.round(bin_source[:-1] / z_delta, decimals=0), dtype='int32') - map_source_size // 4)
    index_upper_source = index_lower_source + map_source_size
    
    # Loop
    label_list = ['ZERO', 'HALF', 'UNITY', 'DOUBLE']
    for label in label_list:
        
        os.makedirs(os.path.join(analyze_folder, '{}/SHIFT/'.format(tag)), exist_ok=True)
        os.makedirs(os.path.join(analyze_folder, '{}/SHIFT/{}'.format(tag, label)), exist_ok=True)
        
        # Info
        with h5py.File(os.path.join(analyze_folder, '{}/INFO/SOM_{}.hdf5'.format(tag, label)), 'r') as file:
            som_realization_lens = file['lens']['realization'][...]
            som_realization_source = file['source']['realization'][...]
        
        with h5py.File(os.path.join(analyze_folder, '{}/INFO/MODEL_{}.hdf5'.format(tag, label)), 'r') as file:
            model_realization_lens = file['lens']['realization'][...]
            model_realization_source = file['source']['realization'][...]
        
        with h5py.File(os.path.join(analyze_folder, '{}/INFO/PRODUCT_{}.hdf5'.format(tag, label)), 'r') as file:
            product_realization_lens = file['lens']['realization'][...]
            product_realization_source = file['source']['realization'][...]
        
        with h5py.File(os.path.join(analyze_folder, '{}/INFO/FIDUCIAL_{}.hdf5'.format(tag, label)), 'r') as file:
            fiducial_realization_lens = file['lens']['realization'][...]
            fiducial_realization_source = file['source']['realization'][...]
        
        with h5py.File(os.path.join(analyze_folder, '{}/INFO/HISTOGRAM_{}.hdf5'.format(tag, label)), 'r') as file:
            histogram_realization_lens = file['lens']['realization'][...]
            histogram_realization_source = file['source']['realization'][...]
        
        data_size, bin_lens_size, z_size = histogram_realization_lens.shape
        data_size, bin_source_size, z_size = histogram_realization_source.shape
        
        # Select
        som_select_lens = numpy.zeros((data_size, bin_lens_size, map_lens_size))
        som_select_source = numpy.zeros((data_size, bin_source_size, map_source_size))
        
        model_select_lens = numpy.zeros((data_size, bin_lens_size, map_lens_size))
        model_select_source = numpy.zeros((data_size, bin_source_size, map_source_size))
        
        product_select_lens = numpy.zeros((data_size, bin_lens_size, map_lens_size))
        product_select_source = numpy.zeros((data_size, bin_source_size, map_source_size))
        
        fiducial_select_lens = numpy.zeros((data_size, bin_lens_size, map_lens_size))
        fiducial_select_source = numpy.zeros((data_size, bin_source_size, map_source_size))
        
        histogram_select_lens = numpy.zeros((data_size, bin_lens_size, map_lens_size))
        histogram_select_source = numpy.zeros((data_size, bin_source_size, map_source_size))
        
        for m in range(bin_lens_size):
            som_select_lens[:, m, :numpy.minimum(map_lens_size, z_size - index_lower_lens[m])] = som_realization_lens[:, m, index_lower_lens[m]: numpy.minimum(z_size, index_upper_lens[m])]
            model_select_lens[:, m, :numpy.minimum(map_lens_size, z_size - index_lower_lens[m])] = model_realization_lens[:, m, index_lower_lens[m]: numpy.minimum(z_size, index_upper_lens[m])]
            product_select_lens[:, m, :numpy.minimum(map_lens_size, z_size - index_lower_lens[m])] = product_realization_lens[:, m, index_lower_lens[m]: numpy.minimum(z_size, index_upper_lens[m])]
            fiducial_select_lens[:, m, :numpy.minimum(map_lens_size, z_size - index_lower_lens[m])] = fiducial_realization_lens[:, m, index_lower_lens[m]: numpy.minimum(z_size, index_upper_lens[m])]
            histogram_select_lens[:, m, :numpy.minimum(map_lens_size, z_size - index_lower_lens[m])] = histogram_realization_lens[:, m, index_lower_lens[m]: numpy.minimum(z_size, index_upper_lens[m])]
        
        for m in range(bin_source_size):
            som_select_source[:, m, :numpy.minimum(map_source_size, z_size - index_lower_source[m])] = som_realization_source[:, m, index_lower_source[m]: numpy.minimum(z_size, index_upper_source[m])]
            model_select_source[:, m, :numpy.minimum(map_source_size, z_size - index_lower_source[m])] = model_realization_source[:, m, index_lower_source[m]: numpy.minimum(z_size, index_upper_source[m])]
            product_select_source[:, m, :numpy.minimum(map_source_size, z_size - index_lower_source[m])] = product_realization_source[:, m, index_lower_source[m]: numpy.minimum(z_size, index_upper_source[m])]
            fiducial_select_source[:, m, :numpy.minimum(map_source_size, z_size - index_lower_source[m])] = fiducial_realization_source[:, m, index_lower_source[m]: numpy.minimum(z_size, index_upper_source[m])]
            histogram_select_source[:, m, :numpy.minimum(map_source_size, z_size - index_lower_source[m])] = histogram_realization_source[:, m, index_lower_source[m]: numpy.minimum(z_size, index_upper_source[m])]
        
        som_covariance_lens = numpy.cov(numpy.reshape(som_select_lens, (data_size, bin_lens_size * map_lens_size)), rowvar=False)
        som_covariance_source = numpy.cov(numpy.reshape(som_select_source, (data_size, bin_source_size * map_source_size)), rowvar=False)
        
        model_covariance_lens = numpy.cov(numpy.reshape(model_select_lens, (data_size, bin_lens_size * map_lens_size)), rowvar=False)
        model_covariance_source = numpy.cov(numpy.reshape(model_select_source, (data_size, bin_source_size * map_source_size)), rowvar=False)
        
        product_covariance_lens = numpy.cov(numpy.reshape(product_select_lens, (data_size, bin_lens_size * map_lens_size)), rowvar=False)
        product_covariance_source = numpy.cov(numpy.reshape(product_select_source, (data_size, bin_source_size * map_source_size)), rowvar=False)
        
        fiducial_covariance_lens = numpy.cov(numpy.reshape(fiducial_select_lens, (data_size, bin_lens_size * map_lens_size)), rowvar=False)
        fiducial_covariance_source = numpy.cov(numpy.reshape(fiducial_select_source, (data_size, bin_source_size * map_source_size)), rowvar=False)
        
        histogram_covariance_lens = numpy.cov(numpy.reshape(histogram_select_lens, (data_size, bin_lens_size * map_lens_size)), rowvar=False)
        histogram_covariance_source = numpy.cov(numpy.reshape(histogram_select_source, (data_size, bin_source_size * map_source_size)), rowvar=False)
        
        # Configuration
        os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
        pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
        pyplot.rcParams['text.usetex'] = True
        pyplot.rcParams['font.size'] = 20
        
        # Plot
        figure = plot_covariance(bin_lens_size, map_lens_size, som_covariance_lens)
        figure.savefig(os.path.join(analyze_folder, '{}/SHIFT/{}/FIGURE_SOM_LENS.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_covariance(bin_source_size, map_source_size, som_covariance_source)
        figure.savefig(os.path.join(analyze_folder, '{}/SHIFT/{}/FIGURE_SOM_SOURCE.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_covariance(bin_lens_size, map_lens_size, model_covariance_lens)
        figure.savefig(os.path.join(analyze_folder, '{}/SHIFT/{}/FIGURE_MODEL_LENS.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_covariance(bin_source_size, map_source_size, model_covariance_source)
        figure.savefig(os.path.join(analyze_folder, '{}/SHIFT/{}/FIGURE_MODEL_SOURCE.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_covariance(bin_lens_size, map_lens_size, product_covariance_lens)
        figure.savefig(os.path.join(analyze_folder, '{}/SHIFT/{}/FIGURE_PRODUCT_LENS.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_covariance(bin_source_size, map_source_size, product_covariance_source)
        figure.savefig(os.path.join(analyze_folder, '{}/SHIFT/{}/FIGURE_PRODUCT_SOURCE.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_covariance(bin_lens_size, map_lens_size, fiducial_covariance_lens)
        figure.savefig(os.path.join(analyze_folder, '{}/SHIFT/{}/FIGURE_FIDUCIAL_LENS.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_covariance(bin_source_size, map_source_size, fiducial_covariance_source)
        figure.savefig(os.path.join(analyze_folder, '{}/SHIFT/{}/FIGURE_FIDUCIAL_SOURCE.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_covariance(bin_lens_size, map_lens_size, histogram_covariance_lens)
        figure.savefig(os.path.join(analyze_folder, '{}/SHIFT/{}/FIGURE_HISTOGRAM_LENS.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_covariance(bin_source_size, map_source_size, histogram_covariance_source)
        figure.savefig(os.path.join(analyze_folder, '{}/SHIFT/{}/FIGURE_HISTOGRAM_SOURCE.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Analysis Shift')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)