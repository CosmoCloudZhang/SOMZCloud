import os
import h5py
import time
import numpy
import argparse
from rail import core
from matplotlib import colors, pyplot
from rail.estimation.algos import somoclu_som

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
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    chunk = 100000
    model_name = os.path.join(som_folder, '{}/INFORM/INFORM.pkl'.format(tag))
    column_list = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_size = len(file['photometry']['redshift'])
        application_mean = numpy.zeros(model['n_rows'] * model['n_columns'])
        application_occupation = numpy.zeros(model['n_rows'] * model['n_columns'])
        
        for m in range(application_size // chunk + 1):
            begin = m * chunk
            stop = min((m + 1) * chunk, application_size)
            
            application = {key: file['photometry'][key][begin: stop].astype(numpy.float32) for key in file['photometry'].keys()}
            column = somoclu_som._computemagcolordata(data=application, mag_column_name='mag_i_lsst', column_names=column_list, colusage='colors')
            
            coordinate = somoclu_som.get_bmus(model['som'], column)
            label = coordinate[:, 0] * model['n_columns'] + coordinate[:, 1]
            
            application_mean += numpy.bincount(label, minlength=model['n_rows'] * model['n_columns'], weights=application['redshift'])
            application_occupation += numpy.bincount(label, minlength=model['n_rows'] * model['n_columns'], weights=numpy.ones(len(label)))
        
        application_mean = numpy.divide(application_mean, application_occupation, out=numpy.ones_like(application_mean) * numpy.nan, where=application_occupation != 0)
        del label, column, coordinate, application, application_size, application_occupation
        
    # Degradation
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        degradation_size = len(file['photometry']['redshift'])
        degradation_mean = numpy.zeros(model['n_rows'] * model['n_columns'])
        degradation_occupation = numpy.zeros(model['n_rows'] * model['n_columns'])
        
        for m in range(degradation_size // chunk + 1):
            begin = m * chunk
            stop = min((m + 1) * chunk, degradation_size)
            
            degradation = {key: file['photometry'][key][begin: stop].astype(numpy.float32) for key in file['photometry'].keys()}
            column = somoclu_som._computemagcolordata(data=degradation, mag_column_name='mag_i_lsst', column_names=column_list, colusage='colors')
            
            coordinate = somoclu_som.get_bmus(model['som'], column)
            label = coordinate[:, 0] * model['n_columns'] + coordinate[:, 1]
            
            degradation_mean += numpy.bincount(label, minlength=model['n_rows'] * model['n_columns'], weights=degradation['redshift'])
            degradation_occupation += numpy.bincount(label, minlength=model['n_rows'] * model['n_columns'], weights=numpy.ones(len(label)))
        
        degradation_mean = numpy.divide(degradation_mean, degradation_occupation, out=numpy.ones_like(degradation_mean) * numpy.nan, where=degradation_occupation != 0)
        del label, column, coordinate, degradation, degradation_size, degradation_occupation
        
    # Augmentation
    with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        augmentation_size = len(file['photometry']['redshift'])
        augmentation_mean = numpy.zeros(model['n_rows'] * model['n_columns'])
        augmentation_occupation = numpy.zeros(model['n_rows'] * model['n_columns'])
        
        for m in range(augmentation_size // chunk + 1):
            begin = m * chunk
            stop = min((m + 1) * chunk, augmentation_size)
            
            augmentation = {key: file['photometry'][key][begin: stop].astype(numpy.float32) for key in file['photometry'].keys()}
            column = somoclu_som._computemagcolordata(data=augmentation, mag_column_name='mag_i_lsst', column_names=column_list, colusage='colors')
            
            coordinate = somoclu_som.get_bmus(model['som'], column)
            label = coordinate[:, 0] * model['n_columns'] + coordinate[:, 1]
            
            augmentation_mean += numpy.bincount(label, minlength=model['n_rows'] * model['n_columns'], weights=augmentation['redshift'])
            augmentation_occupation += numpy.bincount(label, minlength=model['n_rows'] * model['n_columns'], weights=numpy.ones(len(label)))
        
        augmentation_mean = numpy.divide(augmentation_mean, augmentation_occupation, out=numpy.ones_like(augmentation_mean) * numpy.nan, where=augmentation_occupation != 0)
        del label, column, coordinate, augmentation, augmentation_size, augmentation_occupation
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_size = len(file['photometry']['redshift'])
        combination_mean = numpy.zeros(model['n_rows'] * model['n_columns'])
        combination_occupation = numpy.zeros(model['n_rows'] * model['n_columns'])
        
        for m in range(combination_size // chunk + 1):
            begin = m * chunk
            stop = min((m + 1) * chunk, combination_size)
            
            combination = {key: file['photometry'][key][begin: stop].astype(numpy.float32) for key in file['photometry'].keys()}
            column = somoclu_som._computemagcolordata(data=combination, mag_column_name='mag_i_lsst', column_names=column_list, colusage='colors')
            
            coordinate = somoclu_som.get_bmus(model['som'], column)
            label = coordinate[:, 0] * model['n_columns'] + coordinate[:, 1]
            
            combination_mean += numpy.bincount(label, minlength=model['n_rows'] * model['n_columns'], weights=combination['redshift'])
            combination_occupation += numpy.bincount(label, minlength=model['n_rows'] * model['n_columns'], weights=numpy.ones(len(label)))
        
        combination_mean = numpy.divide(combination_mean, combination_occupation, out=numpy.ones_like(combination_mean) * numpy.nan, where=combination_occupation != 0)
        del label, column, coordinate, combination, combination_size, combination_occupation
    
    # Plot
    normalize = colors.Normalize(vmin=0.0, vmax=3.0)
    figure, plot = pyplot.subplots(nrows=2, ncols=2, figsize=(12, 12))
    
    mesh = plot[0, 0].imshow(application_mean.reshape(model['n_rows'], model['n_columns']), norm=normalize, cmap='plasma')
    plot[0, 0].set_title(r'$\mathrm{application}$')
    plot[0, 0].set_xticklabels([])
    plot[0, 0].set_yticklabels([])
    
    mesh = plot[0, 1].imshow(degradation_mean.reshape(model['n_rows'], model['n_columns']), norm=normalize, cmap='plasma')
    plot[0, 1].set_title(r'$\mathrm{degradation}$')
    plot[0, 1].set_xticklabels([])
    plot[0, 1].set_yticklabels([])
    
    mesh = plot[1, 0].imshow(augmentation_mean.reshape(model['n_rows'], model['n_columns']), norm=normalize, cmap='plasma')
    plot[1, 0].set_title(r'$\mathrm{augmentation}$')
    plot[1, 0].set_xticklabels([])
    plot[1, 0].set_yticklabels([])
    
    mesh = plot[1, 1].imshow(combination_mean.reshape(model['n_rows'], model['n_columns']), norm=normalize, cmap='plasma')
    plot[1, 1].set_title(r'$\mathrm{combination}$')
    plot[1, 1].set_xticklabels([])
    plot[1, 1].set_yticklabels([])
    
    color_bar = figure.colorbar(mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    color_bar.set_label(r'$\mathrm{Mean \, Redshift}$')
    figure.subplots_adjust(bottom=0.15)
    
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/SOM/'.format(tag)), exist_ok=True)
    figure.savefig(os.path.join(figure_folder, '{}/SOM/FIGURE{}.png'.format(tag, index)), format='png', bbox_inches='tight', dpi=512)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Dataset')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all suites of datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all suites of datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)