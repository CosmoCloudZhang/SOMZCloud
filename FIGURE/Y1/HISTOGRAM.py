import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot


def main(tag, index, folder):
    '''
    Plot the histogram of the redshifts
    
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
    os.makedirs(os.path.join(figure_folder, '{}/HISTOGRAM/'.format(tag)), exist_ok=True)
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_redshift_true = file['photometry']['redshift_true'][...]
    
    # Degradation
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        degradation_redshift_true = file['photometry']['redshift_true'][...]
    
    # Augmentation
    with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        augmentation_redshift_true = file['photometry']['redshift_true'][...]
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_redshift_true = file['photometry']['redshift_true'][...]
    
    # Bin
    z1 = 0.0
    z2 = 3.0
    z_size = 100
    z_bin = numpy.linspace(z1, z2, z_size + 1)
    
    figure, plot = pyplot.subplots(nrows=1, ncols=1, figsize=(12, 6))
    
    plot.hist(application_redshift_true, bins=z_bin, linewidth=4.0, density=True, histtype='step', color='black', label=r'$\mathtt{Application}$')
    
    plot.hist(degradation_redshift_true, bins=z_bin, linewidth=4.0, density=True, histtype='step', color='darkblue', label=r'$\mathtt{Degradation}$')
    
    plot.hist(augmentation_redshift_true, bins=z_bin, linewidth=4.0, density=True, histtype='step', color='darkred', label=r'$\mathtt{Augmentation}$')
    
    plot.hist(combination_redshift_true, bins=z_bin, linewidth=4.0, density=True, histtype='step', color='darkorange', label=r'$\mathtt{Combination}$')
    
    plot.legend()
    plot.set_xlim(z_bin.min(), z_bin.max())
    
    plot.set_ylabel(r'$\phi (z)$')
    plot.set_xlabel(r'$z_\mathrm{true}$')
    
    # Save
    figure.savefig(os.path.join(figure_folder, '{}/HISTOGRAM/FIGURE{}.pdf'.format(tag, index)), dpi=512, format='pdf', bbox_inches='tight')
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Histogram')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)