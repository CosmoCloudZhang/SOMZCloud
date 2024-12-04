import os
import time
import numpy
import argparse
import matplotlib
from rail import core
import multiprocessing
from matplotlib import cm
from matplotlib import pyplot
from rail.estimation.algos import somoclu_som

def plot(input_data, model_data):
    """
    Plot the som figure.
    
    Arguments:
        input_data (core.data.TableHandle): The input data.
        model_data (core.data.ModelHandle): The model data
    
    Returns:
        matplotlib.figure.Figure: The figure.
    """
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    matplotlib.use('Agg')
    
    model = model_data['som']
    column_list = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    input = somoclu_som._computemagcolordata(data=input_data['photometry'], mag_column_name='mag_i_lsst', column_names=column_list, colusage='colors')
    
    output = somoclu_som.get_bmus(model, input)
    mean_redshift = numpy.zeros((model_data['n_rows'], model_data['n_columns']))
    cell_occupation = numpy.zeros((model_data['n_rows'], model_data['n_columns']))
    
    input_size = input_data['photometry']['redshift'].size
    for k in range(input_size):
        x, y = output[k]
        cell_occupation[x, y] += 1
        mean_redshift[x, y] += input_data['photometry']['redshift'][k]
    
    cell_occupation[cell_occupation == 0] = numpy.nan
    mean_redshift = numpy.divide(mean_redshift, cell_occupation, out=numpy.ones_like(mean_redshift) * numpy.nan, where=cell_occupation != 0)
    
    figure, plot = pyplot.subplots(nrows=1, ncols=2, figsize=(15, 8))
    
    somoclu_som.plot_som(plot[0], cell_occupation.T, grid_type='hexagonal', colormap=cm.viridis, cbar_name=r'$\mathrm{Cell \: Occupation}$')
    
    somoclu_som.plot_som(plot[1], mean_redshift.T, grid_type='hexagonal', colormap=cm.coolwarm, cbar_name=r'$\mathrm{Mean \: Redshift}$')
    
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
    os.makedirs(os.path.join(plot_path, 'SOM/'), exist_ok=True)
    
    model_name = os.path.join(data_path, 'SOM/SOM_INFORM.pkl')
    train_name = os.path.join(data_path, 'SAMPLE/TRAIN_SAMPLE{}.hdf5'.format(index))
    
    train_data = data_store.read_file(key='data', path=train_name, handle_class=core.data.TableHandle)()
    model_data = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    # Plot
    figure = plot(train_data, model_data)
    figure.savefig(os.path.join(plot_path, 'SOM/SOM{}.pdf'.format(index)), bbox_inches='tight')
    pyplot.close(figure)
    
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
    PARSE.add_argument('--length', type=int, required=True, help='The length of the som datasets')
    
    PATH = PARSE.parse_args().path
    NUMBER = PARSE.parse_args().number
    LENGTH = PARSE.parse_args().length
    
    SIZE = LENGTH // NUMBER
    for CHUNK in range(SIZE):
        print('CHUNK: {}'.format(CHUNK + 1))
        with multiprocessing.Pool(processes=NUMBER) as POOL:
            RESULT = POOL.starmap(main, [(PATH, INDEX) for INDEX in range(CHUNK * NUMBER + 1, (CHUNK + 1) * NUMBER + 1)])
