import os
import time
import h5py
import numpy
import argparse
from matplotlib import pyplot
from matplotlib import colors


def main(tag, folder):
    '''
    Plot the shear noise as function of the galaxy sizes and brightnesses
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    figure_folder = os.path.join(folder, 'FIGURE/')
    ensemble_folder = os.path.join(folder, 'ENSEMBLE/')
    
    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/COVARIANCE/'.format(tag)), exist_ok=True)
    
    # Ensemble
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/FZB.hdf5'.format(tag)), 'r') as file:
        fzb_lens = file['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/SOM.hdf5'.format(tag)), 'r') as file:
        som_lens = file['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/HISTOGRAM.hdf5'.format(tag)), 'r') as file:
        histogram_lens = file['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/FZB.hdf5'.format(tag)), 'r') as file:
        fzb_source = file['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/SOM.hdf5'.format(tag)), 'r') as file:
        som_source = file['ensemble'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/HISTOGRAM.hdf5'.format(tag)), 'r') as file:
        histogram_source = file['ensemble'][...]
    
    # Bin
    bin_lens_size = 5
    bin_source_size = 5
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    data_size = grid_size + 1
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # FZB LENS
    fzb_matrix_lens = numpy.cov(numpy.reshape(fzb_lens, (-1, bin_lens_size * data_size)), rowvar=False)
    
    # Plot
    norm = colors.Normalize(vmin = -0.25, vmax = +0.25)
    figure, plot = pyplot.subplots(nrows = bin_lens_size, ncols = bin_lens_size, figsize = (3 * bin_lens_size, 3 * bin_lens_size))
    
    for m in range(bin_lens_size):
        for n in range(bin_lens_size):
            
            map = fzb_matrix_lens[n * data_size: n * data_size + data_size // 2, m * data_size: m * data_size + data_size // 2]
            image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\Delta \mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')
    figure.savefig(os.path.join(figure_folder, '{}/COVARIANCE/FIGURE_FZB_LENS.png'.format(tag)), bbox_inches='tight', format='png', dpi=512)
    
    # HISTOGRAM LENS
    histogram_matrix_lens = numpy.cov(numpy.reshape(histogram_lens, (-1, bin_lens_size * data_size)), rowvar=False)
    
    # Plot
    norm = colors.Normalize(vmin = -0.25, vmax = +0.25)
    figure, plot = pyplot.subplots(nrows = bin_lens_size, ncols = bin_lens_size, figsize = (3 * bin_lens_size, 3 * bin_lens_size))
    
    for m in range(bin_lens_size):
        for n in range(bin_lens_size):
            
            map = histogram_matrix_lens[n * data_size: n * data_size + data_size // 2, m * data_size: m * data_size + data_size // 2]
            image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\Delta \mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')
    figure.savefig(os.path.join(figure_folder, '{}/COVARIANCE/FIGURE_HISTOGRAM_LENS.png'.format(tag)), bbox_inches='tight', format='png', dpi=512)
    
    # SOM LENS
    som_matrix_lens = numpy.cov(numpy.reshape(som_lens, (-1, bin_lens_size * data_size)), rowvar=False)
    
    # Plot
    norm = colors.Normalize(vmin = -0.25, vmax = +0.25)
    figure, plot = pyplot.subplots(nrows = bin_lens_size, ncols = bin_lens_size, figsize = (3 * bin_lens_size, 3 * bin_lens_size))
    
    for m in range(bin_lens_size):
        for n in range(bin_lens_size):
            
            map = som_matrix_lens[n * data_size: n * data_size + data_size // 2, m * data_size: m * data_size + data_size // 2]
            image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\Delta \mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')
    figure.savefig(os.path.join(figure_folder, '{}/COVARIANCE/FIGURE_SOM_LENS.png'.format(tag)), bbox_inches='tight', format='png', dpi=512)
    
    # FZB SOURCE
    fzb_matrix_source = numpy.cov(numpy.reshape(fzb_source, (-1, bin_source_size * data_size)), rowvar=False)
    
    # Plot
    norm = colors.Normalize(vmin = -0.50, vmax = +0.50)
    figure, plot = pyplot.subplots(nrows = bin_source_size, ncols = bin_source_size, figsize = (3 * bin_source_size, 3 * bin_source_size))
    
    for m in range(bin_source_size):
        for n in range(bin_source_size):
            
            map = fzb_matrix_source[n * data_size: n * data_size + data_size // 2, m * data_size: m * data_size + data_size // 2]
            image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\Delta \mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')
    figure.savefig(os.path.join(figure_folder, '{}/COVARIANCE/FIGURE_FZB_SOURCE.png'.format(tag)), bbox_inches='tight', format='png', dpi=512)
    
    # HISTOGRAM SOURCE
    histogram_matrix_source = numpy.cov(numpy.reshape(histogram_source, (-1, bin_source_size * data_size)), rowvar=False)
    
    # Plot
    norm = colors.Normalize(vmin = -0.50, vmax = +0.50)
    figure, plot = pyplot.subplots(nrows = bin_source_size, ncols = bin_source_size, figsize = (3 * bin_source_size, 3 * bin_source_size))
    
    for m in range(bin_source_size):
        for n in range(bin_source_size):
            
            map = histogram_matrix_source[n * data_size: n * data_size + data_size // 2, m * data_size: m * data_size + data_size // 2]
            image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\Delta \mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')
    figure.savefig(os.path.join(figure_folder, '{}/COVARIANCE/FIGURE_HISTOGRAM_SOURCE.png'.format(tag)), bbox_inches='tight', format='png', dpi=512)
    
    # SOM SOURCE
    som_matrix_source = numpy.cov(numpy.reshape(som_source, (-1, bin_source_size * data_size)), rowvar=False)
    
    # Plot
    norm = colors.Normalize(vmin = -0.50, vmax = +0.50)
    figure, plot = pyplot.subplots(nrows = bin_source_size, ncols = bin_source_size, figsize = (3 * bin_source_size, 3 * bin_source_size))
    
    for m in range(bin_source_size):
        for n in range(bin_source_size):
            
            map = som_matrix_source[n * data_size: n * data_size + data_size // 2, m * data_size: m * data_size + data_size // 2]
            image = plot[n, m].imshow(map, norm = norm, cmap = 'coolwarm', origin = 'upper')
            plot[n, m].axis('off')
    
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\Delta \mathrm{Cov} \: [\phi^{m_1} (z_1), \phi^{m_2} (z_2)]$')
    figure.savefig(os.path.join(figure_folder, '{}/COVARIANCE/FIGURE_SOM_SOURCE.png'.format(tag)), bbox_inches='tight', format='png', dpi=512)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Sigma')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)
