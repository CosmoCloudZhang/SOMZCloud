import os
import h5py
import time
import numpy
import argparse
from matplotlib import colors
from matplotlib import pyplot


def main(tag, index, folder):
    '''
    Plot the SOM map of multiple datasets
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    figure_folder = os.path.join(folder, 'FIGURE/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size1 = file['meta']['cell_size1'][...]
        cell_size2 = file['meta']['cell_size2'][...]
        application_cell_count = file['meta']['cell_count'][...]
        application_cell_z_true = file['meta']['cell_z_true'][...]
    application_map = application_cell_z_true.reshape((cell_size1, cell_size2))
    
    # Selection
    with h5py.File(os.path.join(dataset_folder, '{}/SELECTION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size1 = file['meta']['cell_size1'][...]
        cell_size2 = file['meta']['cell_size2'][...]
        selection_cell_count = file['meta']['cell_count'][...]
        selection_cell_z_true = file['meta']['cell_z_true'][...]
    selection_map = selection_cell_z_true.reshape((cell_size1, cell_size2))
    
    # Metric
    cell_ratio = numpy.divide(selection_cell_count / numpy.sum(selection_cell_count), application_cell_count / numpy.sum(application_cell_count), out=numpy.zeros(cell_size1 * cell_size2), where=application_cell_count > 0)
    cell_metric = numpy.sqrt(numpy.mean(numpy.square(cell_ratio - 1)))
    print('Metric: {:.3f}'.format(cell_metric))
    
    # Plot
    normalize = colors.Normalize(vmin=0.0, vmax=2.0)
    figure, plot = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 8))
    
    mesh = plot[0].imshow(application_map, norm=normalize, cmap='coolwarm')
    plot[0].set_title(r'$\mathrm{application}$')
    plot[0].axis('off')
    
    mesh = plot[1].imshow(selection_map, norm=normalize, cmap='coolwarm')
    plot[1].set_title(r'$\mathrm{selection}$')
    plot[1].axis('off')
    
    color_bar = figure.colorbar(mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    figure.subplots_adjust(bottom=0.0, hspace=0.00, wspace=0.05)
    color_bar.set_label(r'$\langle z_\mathrm{true} \rangle$')
    
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/SOM/'.format(tag)), exist_ok=True)
    
    figure.savefig(os.path.join(figure_folder, '{}/SOM/FIGURE{}.pdf'.format(tag, index)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure SOM')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all suites of datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all suites of datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)