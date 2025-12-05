import os
import h5py
import time
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
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/SOM/'.format(tag)), exist_ok=True)
    
    # Plot
    os.environ['PATH'] = '/pscratch/sd/y/yhzhang/texlive/2025/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size1 = file['meta']['cell_size1'][...]
        cell_size2 = file['meta']['cell_size2'][...]
        application_cell_z_true = file['meta']['cell_z_true'][...]
    application_map = application_cell_z_true.reshape((cell_size1, cell_size2))
    
    # Restriction
    with h5py.File(os.path.join(dataset_folder, '{}/RESTRICTION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size1 = file['meta']['cell_size1'][...]
        cell_size2 = file['meta']['cell_size2'][...]
        restriction_cell_z_true = file['meta']['cell_z_true'][...]
    restriction_map = restriction_cell_z_true.reshape((cell_size1, cell_size2))
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size1 = file['meta']['cell_size1'][...]
        cell_size2 = file['meta']['cell_size2'][...]
        combination_cell_z_true = file['meta']['cell_z_true'][...]
    combination_map = combination_cell_z_true.reshape((cell_size1, cell_size2))
    
    # Plot
    normalize = colors.Normalize(vmin=-0.15, vmax=+0.15)
    figure, plot = pyplot.subplots(nrows=2, ncols=1, figsize=(8, 12))
    figure.subplots_adjust(right=0.90, bottom=0.0, hspace=0.15, wspace=0.0)
    
    # Restriction
    mesh1 = plot[0].imshow((restriction_map - application_map) / (1 + application_map), norm=normalize, cmap='coolwarm')
    plot[0].set_title(r'$\mathrm{Restriction}$')
    plot[0].axis('off')
    
    colorbar1 = figure.add_axes([0.90, 0.50, 0.05, 0.35])
    colorbar1 = figure.colorbar(mesh1, cax=colorbar1, orientation='vertical')
    colorbar1.set_label(r'$\delta_{\langle z_\mathrm{true} \rangle}$')
    
    # Combination
    mesh2 = plot[1].imshow((combination_map - application_map) / (1 + application_map), norm=normalize, cmap='coolwarm')
    plot[1].set_title(r'$\mathrm{Combination}$')
    plot[1].axis('off')
    
    colorbar2 = figure.add_axes([0.90, 0.02, 0.05, 0.35])
    colorbar2 = figure.colorbar(mesh2, cax=colorbar2, orientation='vertical')
    colorbar2.set_label(r'$\delta_{\langle z_\mathrm{true} \rangle}$')
    
    # Save
    figure.savefig(os.path.join(figure_folder, '{}/SOM/FIGURE{}.pdf'.format(tag, index)), dpi=512, format='pdf', bbox_inches='tight')
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