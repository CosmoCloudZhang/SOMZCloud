import os
import h5py
import time
import numpy
import argparse
from matplotlib import colors
from matplotlib import pyplot
from matplotlib.gridspec import GridSpec


def main(tag, index, folder):
    '''
    Plot the figure of the optical color-color diagram
    
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
    os.makedirs(os.path.join(figure_folder, '{}/INFRARED/'.format(tag)), exist_ok=True)
    
    # Plot
    os.environ['PATH'] = '/pscratch/sd/y/yhzhang/texlive/2025/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_color1 = file['photometry']['mag_i_lsst'][...] - file['photometry']['mag_z_lsst'][...]
        application_color2 = file['photometry']['mag_z_lsst'][...] - file['photometry']['mag_y_lsst'][...]
    
    # Degradation
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        degradation_color1 = file['photometry']['mag_i_lsst'][...] - file['photometry']['mag_z_lsst'][...]
        degradation_color2 = file['photometry']['mag_z_lsst'][...] - file['photometry']['mag_y_lsst'][...]
    
    # Augmentation
    with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        augmentation_color1 = file['photometry']['mag_i_lsst'][...] - file['photometry']['mag_z_lsst'][...]
        augmentation_color2 = file['photometry']['mag_z_lsst'][...] - file['photometry']['mag_y_lsst'][...]
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_color1 = file['photometry']['mag_i_lsst'][...] - file['photometry']['mag_z_lsst'][...]
        combination_color2 = file['photometry']['mag_z_lsst'][...] - file['photometry']['mag_y_lsst'][...]
    
    # Bin
    color_size = 100
    
    color1_edge1 = -1.5
    color1_edge2 = +1.5
    color1_bin = numpy.linspace(color1_edge1, color1_edge2, color_size + 1)
    
    color2_edge1 = -1.8
    color2_edge2 = +2.0
    color2_bin = numpy.linspace(color2_edge1, color2_edge2, color_size + 1)
    
    figure = pyplot.figure(figsize = (12, 16))
    normalize = colors.LogNorm(vmin=1, vmax=10000)
    gridspec = GridSpec(nrows=2, ncols=2, figure=figure, top=0.70, bottom=0.15, hspace=0.0, wspace=0.0)
    
    plot = figure.add_subplot(gridspec[0, 0])
    
    plot.text(-1.5, -1.2, r'$\mathtt{Application}$')
    
    image = plot.hist2d(application_color2, application_color1, bins=[color1_bin, color2_bin], norm=normalize, cmap='inferno')[-1]
    
    plot.set_ylabel(r'$i - z$')
    plot.set_xticklabels([])
    
    plot.set_ylim(color1_edge1, color1_edge2)
    plot.set_xlim(color2_edge1, color2_edge2)
    
    plot = figure.add_subplot(gridspec[0, 1])
    
    plot.text(-1.5, -1.2, r'$\mathtt{Degradation}$')
    
    image = plot.hist2d(degradation_color2, degradation_color1, bins=[color1_bin, color2_bin], norm=normalize, cmap='inferno')[-1]
    
    plot.set_yticklabels([])
    plot.set_xticklabels([])
    
    plot.set_ylim(color1_edge1, color1_edge2)
    plot.set_xlim(color2_edge1, color2_edge2)
    
    plot = figure.add_subplot(gridspec[1, 0])
    
    plot.text(-1.5, -1.2, r'$\mathtt{Augmentation}$')
    
    image = plot.hist2d(augmentation_color2, augmentation_color1, bins=[color1_bin, color2_bin], norm=normalize, cmap='inferno')[-1]
    
    plot.set_ylim(color1_edge1, color1_edge2)
    plot.set_xlim(color2_edge1, color2_edge2)
    
    plot.set_ylabel(r'$i - z$')
    plot.set_xlabel(r'$z - y$')
    
    plot = figure.add_subplot(gridspec[1, 1])
    
    plot.text(-1.5, -1.2, r'$\mathtt{Combination}$')
    
    image = plot.hist2d(combination_color2, combination_color1, bins=[color1_bin, color2_bin], norm=normalize, cmap='inferno')[-1]
    
    plot.set_ylim(color1_edge1, color1_edge2)
    plot.set_xlim(color2_edge1, color2_edge2)
    
    plot.set_yticklabels([])
    plot.set_xlabel(r'$z - y$')
    
    color_bar = figure.colorbar(image, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    color_bar.set_label(r'$\mathrm{Counts}$')
    figure.subplots_adjust(bottom=0.15)
    
    # Save    
    figure.savefig(os.path.join(figure_folder, '{}/INFRARED/FIGURE{}.pdf'.format(tag, index)), dpi=512, format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Infrared')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)