import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot, colors


def main(tag, index, folder):
    '''
    Plot the figures of the redshift estimation
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        float: The duration of the plotter
    '''
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    figure_folder = os.path.join(folder, 'FIGURE/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/SAMPLE/'.format(tag)), exist_ok=True)
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    
    z1_source = 0.05
    z2_source = 2.95
    
    z1 = 0.0
    z2 = 3.0
    
    bin_size = 100
    z_bin = numpy.linspace(z1, z2, bin_size + 1)
    
    # Magnitude
    magnitude1 = 16.0
    magnitude2 = 26.0
    magnitude_grid = numpy.linspace(magnitude1, magnitude2, bin_size + 1)
    
    # Values
    slope = 4
    intercept = 18.0
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_magnitude = file['photometry']['mag_i_lsst'][...]
    
    # Estimator
    with h5py.File(os.path.join(model_folder, '{}/TARGET/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        z_phot = file['z_phot'][...]
    
    # Configuration
    os.environ['PATH'] = '/pscratch/sd/y/yhzhang/texlive/2025/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    normalize = colors.LogNorm(vmin=1, vmax=10000)
    figure, plot = pyplot.subplots(nrows=1, ncols=1, figsize=(12, 8))
    
    image = plot.hist2d(x=z_phot, y=application_magnitude, bins=[z_bin, magnitude_grid], norm=normalize, cmap='plasma')[-1]
    
    plot.plot(numpy.ones(bin_size) * z1_source, numpy.linspace(magnitude1, magnitude2, bin_size), color='black', linestyle='--', linewidth=2.5)
    
    plot.plot(numpy.ones(bin_size) * z2_source, numpy.linspace(magnitude1, magnitude2, bin_size), color='black', linestyle='--', linewidth=2.5)
    
    plot.plot(numpy.ones(bin_size) * z1_lens, numpy.linspace(magnitude1, slope * z1_lens + intercept, bin_size), color='black', linestyle='-.', linewidth=2.5)
    
    plot.plot(numpy.ones(bin_size) * z2_lens, numpy.linspace(magnitude1, slope * z2_lens + intercept, bin_size), color='black', linestyle='-.', linewidth=2.5)
    
    plot.plot(numpy.linspace(z1_lens, z2_lens, bin_size + 1), slope * numpy.linspace(z1_lens, z2_lens, bin_size + 1) + intercept, color='black', linestyle='-.', linewidth=2.5)
    
    plot.set_xlim(z1, z2)
    plot.set_ylim(magnitude1, magnitude2)
    
    plot.set_ylabel(r'$i$')
    plot.set_xlabel(r'$z_\mathrm{phot}$')
    
    # Color bar
    color_bar = figure.colorbar(image, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    figure.subplots_adjust(bottom=0.20, wspace=0.00, hspace=0.00)
    color_bar.set_label(r'$\mathrm{Counts}$')
    
    # Save
    figure.savefig(os.path.join(figure_folder, '{}/SAMPLE/FIGURE{}.pdf'.format(tag, index)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Sample')
    PARSE.add_argument('--tag', type=str, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)