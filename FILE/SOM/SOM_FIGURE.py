import os
import time
import numpy
import argparse
from rail import core
import multiprocessing
from matplotlib import cm
from matplotlib import pyplot
from rail.estimation.algos import somoclu_som


def plot(train_data, model_data):
    """
    Plot the som figure.
    
    Arguments:
        train_data (core.data.TableHandle): The training data.
        model_data (core.data.ModelHandle): The model data.
    
    Returns:
        matplotlib.figure.Figure: The figure.
    """
    
    band_list = ['u', 'g', 'r', 'i', 'z', 'y']
    band_name = ['mag_{}_lsst'.format(band) for band in band_list]
    
    size = train_data()['photometry']['redshift'].size
    data = numpy.zeros((size, len(band_list)))
    for k in range(len(band_list)):
        data[:, k] = train_data()['photometry'][band_name[k]]
    
    som = model_data()['som']
    coordinates = somoclu_som.get_bmus(som, data)
    
    mean_redshift = numpy.zeros((model_data()['n_rows'], model_data()['n_columns']))
    cell_occupation = numpy.zeros((model_data()['n_rows'], model_data()['n_columns']))
    
    for k in range(size):
        x, y = coordinates[k]
        cell_occupation[x, y] += 1
        mean_redshift[x, y] += train_data()['photometry']['redshift'][k]
    
    cell_occupation[cell_occupation == 0] = numpy.nan
    mean_redshift = numpy.divide(mean_redshift, cell_occupation, out=numpy.ones_like(mean_redshift) * numpy.nan, where=cell_occupation != 0)
    
    figure, plot = pyplot.subplots(nrows=1, ncols=2, figsize=(16, 8))
    
    somoclu_som.plot_som(plot[0], cell_occupation.T, grid_type='hexagonal', colormap=cm.coolwarm, cbar_name='Cell Occupation')
    
    somoclu_som.plot_som(plot[1], mean_redshift.T, grid_type='hexagonal', colormap=cm.coolwarm, cbar_name='Mean Redshift')
    
    return figure


def main(path, index):
    """
    The main function to plot the som figure.
    
    Arguments:
        path (str): The path to the base folder.
        index (int): The index of the datasets.
    
    Returns:
        float: The duration of the plotting.
    """
    
    # Data store
    start = time.time()
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Data
    data_path = os.path.join(path, 'DATA/')
    plot_path = os.path.join(path, 'PLOT/')
    
    model_name = os.path.join(data_path, 'SOM/SOM_INFORM{}.pkl'.format(index))
    train_name = os.path.join(data_path, 'SAMPLE/TRAIN_SAMPLE{}.hdf5'.format(index))
    
    train_data = data_store.read_file(key='train', path=train_name, handle_class=core.data.TableHandle)
    model_data = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)
    
    # Plot
    figure = plot(train_data, model_data)
    os.makedirs(os.path.join(plot_path, 'SOM/'), exist_ok=True)
    figure.savefig(os.path.join(plot_path, 'SOM/SOM{}.pdf'.format(index)), bbox_inches='tight')
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    return duration


if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='SOM FIGURE')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the processes')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets')
    
    PATH = PARSE.parse_args().path
    NUMBER = PARSE.parse_args().number
    LENGTH = PARSE.parse_args().length
    
    # Multiprocessing
    SIZE = LENGTH // NUMBER
    for CHUNK in range(SIZE):
        print('CHUNK: {}'.format(CHUNK + 1))
        with multiprocessing.Pool(processes=NUMBER) as POOL:
            POOL.starmap(main, [(PATH, INDEX) for INDEX in range(CHUNK * NUMBER + 1, (CHUNK + 1) * NUMBER + 1)])