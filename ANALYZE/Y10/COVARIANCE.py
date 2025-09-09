import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot
from matplotlib import colors


def plot_covariance(bin_size, map_size, covariance):
    '''
    Plot the covariance matrix
    
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
            
            plot[n, m].set_xticks([])
            plot[n, m].set_yticks([])
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.0, hspace = 0.0)
    
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\mathcal{C} \: [\phi^{a} (z_1), \phi^{b} (z_2)]$')
    return figure


def main(tag, rank, label, folder):
    '''
    Plot the covariance matrix
    
    Arguments:
        tag (str): The tag of the configuration
        rank (str): The rank of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    print('Rank: {}, Label: {}'.format(rank, label))
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    
    os.makedirs(os.path.join(analyze_folder, '{}/COVARIANCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(analyze_folder, '{}/COVARIANCE/{}'.format(tag, label)), exist_ok=True)
    
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
    
    # Synthesize Data
    with h5py.File(os.path.join(synthesize_folder, '{}/{}_{}.hdf5'.format(tag, rank, label)), 'r') as file:
        data_lens = file['lens']['data'][...]
        data_source = file['source']['data'][...]
    
    data_size, bin_lens_size, z_size = data_lens.shape
    data_size, bin_source_size, z_size = data_source.shape
    
    # Select
    select_lens = numpy.zeros((data_size, bin_lens_size, map_lens_size))
    select_source = numpy.zeros((data_size, bin_source_size, map_source_size))
    
    for m in range(bin_lens_size):
        select_lens[:, m, :numpy.minimum(map_lens_size, z_size - index_lower_lens[m])] = data_lens[:, m, index_lower_lens[m]: numpy.minimum(z_size, index_upper_lens[m])]
    
    for m in range(bin_source_size):
        select_source[:, m, :numpy.minimum(map_source_size, z_size - index_lower_source[m])] = data_source[:, m, index_lower_source[m]: numpy.minimum(z_size, index_upper_source[m])]
    
    covariance_lens = numpy.cov(numpy.reshape(select_lens, (data_size, bin_lens_size * map_lens_size)), rowvar=False)
    covariance_source = numpy.cov(numpy.reshape(select_source, (data_size, bin_source_size * map_source_size)), rowvar=False)
    
    # Synthesize Shift
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/{}_{}.hdf5'.format(tag, rank, label)), 'r') as file:
        shift_lens = file['lens']['shift'][...]
        shift_source = file['source']['shift'][...]
    
    data_size, bin_lens_size, z_size = shift_lens.shape
    data_size, bin_source_size, z_size = shift_source.shape
    
    # Select
    select_shift_lens = numpy.zeros((data_size, bin_lens_size, map_lens_size))
    select_shift_source = numpy.zeros((data_size, bin_source_size, map_source_size))
    
    for m in range(bin_lens_size):
        select_shift_lens[:, m, :numpy.minimum(map_lens_size, z_size - index_lower_lens[m])] = shift_lens[:, m, index_lower_lens[m]: numpy.minimum(z_size, index_upper_lens[m])]
    
    for m in range(bin_source_size):
        select_shift_source[:, m, :numpy.minimum(map_source_size, z_size - index_lower_source[m])] = shift_source[:, m, index_lower_source[m]: numpy.minimum(z_size, index_upper_source[m])]
    
    covariance_shift_lens = numpy.cov(numpy.reshape(select_shift_lens, (data_size, bin_lens_size * map_lens_size)), rowvar=False)
    covariance_shift_source = numpy.cov(numpy.reshape(select_shift_source, (data_size, bin_source_size * map_source_size)), rowvar=False)
    
    # Synthesize Scale
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/{}_{}.hdf5'.format(tag, rank, label)), 'r') as file:
        scale_lens = file['lens']['scale'][...]
        scale_source = file['source']['scale'][...]
    
    data_size, bin_lens_size, z_size = scale_lens.shape
    data_size, bin_source_size, z_size = scale_source.shape
    
    # Select
    select_scale_lens = numpy.zeros((data_size, bin_lens_size, map_lens_size))
    select_scale_source = numpy.zeros((data_size, bin_source_size, map_source_size))
    
    for m in range(bin_lens_size):
        select_scale_lens[:, m, :numpy.minimum(map_lens_size, z_size - index_lower_lens[m])] = scale_lens[:, m, index_lower_lens[m]: numpy.minimum(z_size, index_upper_lens[m])]
    
    for m in range(bin_source_size):
        select_scale_source[:, m, :numpy.minimum(map_source_size, z_size - index_lower_source[m])] = scale_source[:, m, index_lower_source[m]: numpy.minimum(z_size, index_upper_source[m])]
    
    covariance_scale_lens = numpy.cov(numpy.reshape(select_scale_lens, (data_size, bin_lens_size * map_lens_size)), rowvar=False)
    covariance_scale_source = numpy.cov(numpy.reshape(select_scale_source, (data_size, bin_source_size * map_source_size)), rowvar=False)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot Data
    figure = plot_covariance(bin_lens_size, map_lens_size, covariance_lens)
    figure.savefig(os.path.join(analyze_folder, '{}/COVARIANCE/{}/FIGURE_{}_LENS.pdf'.format(tag, label, rank)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    figure = plot_covariance(bin_source_size, map_source_size, covariance_source)
    figure.savefig(os.path.join(analyze_folder, '{}/COVARIANCE/{}/FIGURE_{}_SOURCE.pdf'.format(tag, label, rank)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Plot Shift
    figure = plot_covariance(bin_lens_size, map_lens_size, covariance_shift_lens)
    figure.savefig(os.path.join(analyze_folder, '{}/COVARIANCE/{}/FIGURE_{}_SHIFT_LENS.pdf'.format(tag, label, rank)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    figure = plot_covariance(bin_source_size, map_source_size, covariance_shift_source)
    figure.savefig(os.path.join(analyze_folder, '{}/COVARIANCE/{}/FIGURE_{}_SHIFT_SOURCE.pdf'.format(tag, label, rank)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Plot Scale
    figure = plot_covariance(bin_lens_size, map_lens_size, covariance_scale_lens)
    figure.savefig(os.path.join(analyze_folder, '{}/COVARIANCE/{}/FIGURE_{}_SCALE_LENS.pdf'.format(tag, label, rank)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    figure = plot_covariance(bin_source_size, map_source_size, covariance_scale_source)
    figure.savefig(os.path.join(analyze_folder, '{}/COVARIANCE/{}/FIGURE_{}_SCALE_SOURCE.pdf'.format(tag, label, rank)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Analysis Covariance')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--rank', type=str, required=True, help='The rank of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    RANK = PARSE.parse_args().rank
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, RANK, LABEL, FOLDER)