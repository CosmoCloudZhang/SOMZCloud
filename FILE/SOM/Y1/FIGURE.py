import os
import time
import numpy
import argparse
from rail import core
from matplotlib import cm, pyplot
from rail.estimation.algos import somoclu_som


def main(index, folder):
    '''
    Plot the figure of SOM modelling.
    
    Arguments:
        index (int): The number of datasets.
        folder (str): The base folder of the datasets.
    
    Returns:
        float: The duration of the plotter.
    '''
    # Start
    start = time.time()
    print('Index:{}'.format(index))
    
    # Path
    som_folder = os.path.join(folder, 'SOM/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(som_folder, 'FIGURE/'), exist_ok=True)
    
    # Load
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Load
    data_name = os.path.join(dataset_folder, 'COMBINATION/DATA{}.hdf5'.format(index))
    model_name = os.path.join(som_folder, 'INFORM/INFORM.pkl')
    
    input_data = data_store.read_file(key='data', path=data_name, handle_class=core.data.TableHandle)()['photometry']
    input_model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    # SOM
    model = input_model['som']
    column_list = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    input = somoclu_som._computemagcolordata(data=input_data, mag_column_name='mag_i_lsst', column_names=column_list, colusage='magandcolors')
    
    output = somoclu_som.get_bmus(model, input)
    mean_redshift = numpy.zeros((input_model['n_rows'], input_model['n_columns']))
    cell_occupation = numpy.zeros((input_model['n_rows'], input_model['n_columns']))
    
    input_size = input_data['redshift'].size
    for k in range(input_size):
        x, y = output[k]
        cell_occupation[x, y] += 1
        mean_redshift[x, y] += input_data['redshift'][k]
    
    cell_occupation[cell_occupation == 0] = numpy.nan
    mean_redshift = numpy.divide(mean_redshift, cell_occupation, out=numpy.ones_like(mean_redshift) * numpy.nan, where=cell_occupation != 0)
    
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    figure, plot = pyplot.subplots(nrows=1, ncols=2, figsize=(15, 8))
    
    somoclu_som.plot_som(plot[0], cell_occupation.T, grid_type='hexagonal', colormap=cm.viridis, cbar_name=r'$\mathrm{Cell \: Occupation}$')
    
    somoclu_som.plot_som(plot[1], mean_redshift.T, grid_type='hexagonal', colormap=cm.coolwarm, cbar_name=r'$\mathrm{Mean \: Redshift}$')
    
    figure.savefig(os.path.join(som_folder, 'FIGURE/FIGURE{}.png'.format(index)), bbox_inches='tight', dpi=512)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='SOM Figure')
    PARSE.add_argument('--index', type=int, help='The index of the dataset')
    PARSE.add_argument('--folder', type=str, help='The path to the base folder')
    
    # Parse
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(INDEX, FOLDER)