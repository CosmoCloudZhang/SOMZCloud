import os
import h5py
import time
import numpy
import argparse
from rail import core
from matplotlib import colors, pyplot

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
    som_folder = os.path.join(folder, 'SOM/')
    figure_folder = os.path.join(folder, 'FIGURE/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(som_folder, '{}/INFORM/INFORM.pkl'.format(tag))
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_mean = file['meta']['mean'][:].astype(numpy.float32)
    application_map = application_mean.reshape(model['n_rows'], model['n_columns'])
    
    # Degradation
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        degradation_mean = file['meta']['mean'][:].astype(numpy.float32)
    degradation_map = degradation_mean.reshape(model['n_rows'], model['n_columns'])
    
    # Augmentation
    with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        augmentation_mean = file['meta']['mean'][:].astype(numpy.float32)
    augmentation_map = augmentation_mean.reshape(model['n_rows'], model['n_columns'])
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_mean = file['meta']['mean'][:].astype(numpy.float32)
    combination_map = combination_mean.reshape(model['n_rows'], model['n_columns'])
    
    # Plot
    normalize = colors.Normalize(vmin=0.0, vmax=2.0)
    figure, plot = pyplot.subplots(nrows=2, ncols=2, figsize=(12, 12))
    
    mesh = plot[0, 0].imshow(application_map, norm=normalize, cmap='coolwarm')
    plot[0, 0].set_title(r'$\mathrm{application}$')
    plot[0, 0].axis('off')
    
    mesh = plot[0, 1].imshow(degradation_map, norm=normalize, cmap='coolwarm')
    plot[0, 1].set_title(r'$\mathrm{degradation}$')
    plot[0, 1].axis('off')
    
    mesh = plot[1, 0].imshow(augmentation_map, norm=normalize, cmap='coolwarm')
    plot[1, 0].set_title(r'$\mathrm{augmentation}$')
    plot[1, 0].axis('off')
    
    mesh = plot[1, 1].imshow(combination_map, norm=normalize, cmap='coolwarm')
    plot[1, 1].set_title(r'$\mathrm{combination}$')
    plot[1, 1].axis('off')
    
    color_bar = figure.colorbar(mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    figure.subplots_adjust(bottom=0.12, hspace=0.10, wspace=0.05)
    color_bar.set_label(r'$\mathrm{Mean \, Redshift}$')
    
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/MAP/'.format(tag)), exist_ok=True)
    
    figure.savefig(os.path.join(figure_folder, '{}/MAP/FIGURE{}.png'.format(tag, index)), format='png', bbox_inches='tight', dpi=512)
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
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