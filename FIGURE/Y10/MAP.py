import os
import h5py
import time
import argparse
from matplotlib import colors
from matplotlib import pyplot


def main(tag, index, folder):
    '''
    Plot the figure of the datasets
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all suites of datasets
        folder (str): The base folder of all suites of datasets
    
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
    os.makedirs(os.path.join(figure_folder, '{}/MAP/'.format(tag)), exist_ok=True)
    
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
    
    # Degradation
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size1 = file['meta']['cell_size1'][...]
        cell_size2 = file['meta']['cell_size2'][...]
        degradation_cell_z_true = file['meta']['cell_z_true'][...]
    degradation_map = degradation_cell_z_true.reshape((cell_size1, cell_size2))
    
    # Augmentation
    with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size1 = file['meta']['cell_size1'][...]
        cell_size2 = file['meta']['cell_size2'][...]
        augmentation_cell_z_true = file['meta']['cell_z_true'][...]
    augmentation_map = augmentation_cell_z_true.reshape((cell_size1, cell_size2))
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size1 = file['meta']['cell_size1'][...]
        cell_size2 = file['meta']['cell_size2'][...]
        combination_cell_z_true = file['meta']['cell_z_true'][...]
    combination_map = combination_cell_z_true.reshape((cell_size1, cell_size2))
    
    # Plot
    normalize = colors.Normalize(vmin=0.0, vmax=2.0)
    figure, plot = pyplot.subplots(nrows=2, ncols=2, figsize=(12, 12))
    
    mesh = plot[0, 0].imshow(application_map, norm=normalize, cmap='coolwarm')
    plot[0, 0].set_title(r'$\mathtt{Application}$')
    plot[0, 0].axis('off')
    
    mesh = plot[0, 1].imshow(degradation_map, norm=normalize, cmap='coolwarm')
    plot[0, 1].set_title(r'$\mathtt{Degradation}$')
    plot[0, 1].axis('off')
    
    mesh = plot[1, 0].imshow(augmentation_map, norm=normalize, cmap='coolwarm')
    plot[1, 0].set_title(r'$\mathtt{Augmentation}$')
    plot[1, 0].axis('off')
    
    mesh = plot[1, 1].imshow(combination_map, norm=normalize, cmap='coolwarm')
    plot[1, 1].set_title(r'$\mathtt{Combination}$')
    plot[1, 1].axis('off')
    
    color_bar = figure.colorbar(mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    figure.subplots_adjust(bottom=0.12, hspace=0.10, wspace=0.05)
    color_bar.set_label(r'$\langle z_\mathrm{true} \rangle$')
    
    # Save    
    figure.savefig(os.path.join(figure_folder, '{}/MAP/FIGURE{}.pdf'.format(tag, index)), dpi=512, format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Map')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all suites of datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all suites of datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)