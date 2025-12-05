import os
import h5py
import time
import numpy
import argparse
from matplotlib import colors, pyplot
from matplotlib.gridspec import GridSpec


def main(tag, index, folder):
    '''
    Plot the figure of the color-magnitude diagram
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    figure_folder = os.path.join(folder, 'FIGURE/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/DIAGRAM/'.format(tag)), exist_ok=True)
    
    # Plot
    os.environ['PATH'] = '/pscratch/sd/y/yhzhang/texlive/2025/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_magnitude = file['photometry']['mag_i_lsst'][...]
        application_color = file['photometry']['mag_g_lsst'][...] - file['photometry']['mag_z_lsst'][...]
    
    # Degradation
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        degradation_magnitude = file['photometry']['mag_i_lsst'][...]
        degradation_color = file['photometry']['mag_g_lsst'][...] - file['photometry']['mag_z_lsst'][...]
    
    # Augmentation
    with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        augmentation_magnitude = file['photometry']['mag_i_lsst'][...]
        augmentation_color = file['photometry']['mag_g_lsst'][...] - file['photometry']['mag_z_lsst'][...]
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_magnitude = file['photometry']['mag_i_lsst'][...]
        combination_color = file['photometry']['mag_g_lsst'][...] - file['photometry']['mag_z_lsst'][...]
    
    # Bin
    magnitude1 = 16.0
    magnitude2 = 26.0
    magnitude_size = 100
    magnitude_bin = numpy.linspace(magnitude1, magnitude2, magnitude_size + 1)
    
    color1 = -2.0
    color2 = +6.0
    color_size = 100
    color_bin = numpy.linspace(color1, color2, color_size + 1)
    
    figure = pyplot.figure(figsize = (12, 16))
    normalize = colors.LogNorm(vmin=1, vmax=10000)
    gridspec = GridSpec(nrows=2, ncols=2, figure=figure, top=0.70, bottom=0.15, hspace=0.0, wspace=0.0)
    
    plot = figure.add_subplot(gridspec[0, 0])
    
    plot.text(2.1, 17.0, r'$\mathtt{Application}$')
    
    image = plot.hist2d(application_color, application_magnitude, bins=[color_bin, magnitude_bin], norm=normalize, cmap='magma')[-1]
    
    plot.set_ylabel(r'$i$')
    plot.set_xticklabels([])
    
    plot.get_yticklabels()[0].set_visible(False)
    plot.get_xticklabels()[0].set_visible(False)
    
    plot.set_xlim(color_bin.min(), color_bin.max())
    plot.set_ylim(magnitude_bin.min(), magnitude_bin.max())
    
    plot = figure.add_subplot(gridspec[0, 1])
    
    plot.text(2.1, 17.0, r'$\mathtt{Degradation}$')
    
    image = plot.hist2d(degradation_color, degradation_magnitude, bins=[color_bin, magnitude_bin], norm=normalize, cmap='magma')[-1]
    
    plot.set_yticklabels([])
    plot.set_xticklabels([])
    
    plot.get_yticklabels()[0].set_visible(False)
    plot.get_xticklabels()[0].set_visible(False)
    
    plot.set_xlim(color_bin.min(), color_bin.max())
    plot.set_ylim(magnitude_bin.min(), magnitude_bin.max())
    
    plot = figure.add_subplot(gridspec[1, 0])
    
    plot.text(2.1, 17.0, r'$\mathtt{Augmentation}$')
    
    image = plot.hist2d(augmentation_color, augmentation_magnitude, bins=[color_bin, magnitude_bin], norm=normalize, cmap='magma')[-1]
    
    plot.set_xlim(color_bin.min(), color_bin.max())
    plot.set_ylim(magnitude_bin.min(), magnitude_bin.max())
    
    plot.get_yticklabels()[0].set_visible(False)
    plot.get_xticklabels()[0].set_visible(False)
    
    plot.set_ylabel(r'$i$')
    plot.set_xlabel(r'$g - z$')
    
    plot = figure.add_subplot(gridspec[1, 1])
    
    plot.text(2.1, 17.0, r'$\mathtt{Combination}$')
    
    image = plot.hist2d(combination_color, combination_magnitude, bins=[color_bin, magnitude_bin], norm=normalize, cmap='magma')[-1]
    
    plot.set_xlim(color_bin.min(), color_bin.max())
    plot.set_ylim(magnitude_bin.min(), magnitude_bin.max())
    
    plot.get_yticklabels()[0].set_visible(False)
    plot.get_xticklabels()[0].set_visible(False)
    
    plot.set_yticklabels([])
    plot.set_xlabel(r'$g - z$')
    
    color_bar = figure.colorbar(image, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    color_bar.set_label(r'$\mathrm{Counts}$')
    figure.subplots_adjust(bottom=0.15)
    
    # Save
    figure.savefig(os.path.join(figure_folder, '{}/DIAGRAM/FIGURE{}.pdf'.format(tag, index)), dpi=512, format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Diagram')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)