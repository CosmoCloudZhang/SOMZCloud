import os
import h5py
import time
import numpy
import scipy
import argparse
from rail import core
import multiprocessing
from matplotlib import pyplot
from matplotlib import colors
from matplotlib.gridspec import GridSpec


def select(bin_datasets, input_datasets, augment_datasets):
    """
    Select the data.
    
    Arguments:
        bin_datasets (list): The bin datasets.
        input_datasets (list): The input datasets.
        augment_datasets (list): The augment datasets.
    
    Returns:
        numpy.ndarray: The select data.
    """
    z_bin, mag_bin, color_bin = bin_datasets
    z_input, mag_input, color_input = input_datasets
    z_augment, mag_augment, color_augment = augment_datasets
    
    z1 = z_bin.min()
    z2 = z_bin.max()
    z_delta = (z2 - z1) / len(z_bin)
    z_data = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, len(z_bin) - 1)
    
    mag1 = mag_bin.min()
    mag2 = mag_bin.max()
    mag_delta = (mag2 - mag1) / len(mag_bin)
    mag_data = numpy.linspace(mag1 + mag_delta / 2, mag2 - mag_delta / 2, len(mag_bin) - 1)
    
    color1 = color_bin.min()
    color2 = color_bin.max()
    color_delta = (color2 - color1) / len(color_bin)
    color_data = numpy.linspace(color1 + color_delta / 2, color2 - color_delta / 2, len(color_bin) - 1)
    
    pdf, edges = numpy.histogramdd([z_input, mag_input, color_input], bins=[z_bin, mag_bin, color_bin], density=True)
    
    sigma = numpy.quantile(pdf[pdf > 0], 0.01)
    factor = numpy.log(1 + numpy.exp(- numpy.square(pdf / sigma)))
    
    weight = scipy.interpolate.interpn(points=(z_data, mag_data, color_data), values=factor, xi=(z_augment, mag_augment, color_augment), method='linear', bounds_error=False, fill_value=0.0)
    weight = weight / numpy.sum(weight)
    
    count = len(z_input) // 3
    index = numpy.arange(len(z_augment))
    index_sample = numpy.random.choice(index, size=count, replace=True, p=weight)
    
    select_sample = numpy.zeros_like(z_augment, dtype=bool)
    select_sample[index_sample] = True
    return select_sample


def augment(input_data, augment_data, select_data):
    """
    Augment the data.
    
    Arguments:
        data_store (core.data.DataStore): The data store.   
        input_data (core.data.TableHandle): The input data. 
        augment_data (core.data.TableHandle): The augment data.
        select_data (numpy.ndarray): The select data.
    
    Returns:
        core.data.TableHandle: The train data.
    """
    train_data = {}
    for key in input_data['photometry'].keys():
        train_data[key] = numpy.concatenate([input_data['photometry'][key], augment_data['photometry'][key][select_data]])
    return train_data


def plot(bin_datasets, test_datasets, input_datasets, train_datasets, augmented_datasets):
    """
    Plot the data.
    
    Arguments:
        bin_datasets (list): The bin datasets.
        test_datasets (list): The test datasets.
        input_datasets (list): The input datasets.
        train_datasets (list): The train datasets.
        augmented_datasets (list): The augment datasets.
    
    Returns:
        matplotlib.figure.Figure: The figure.
    """
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 15
    
    z_bin, mag_bin, color_bin = bin_datasets
    z_test, mag_test, color_test = test_datasets
    z_input, mag_input, color_input = input_datasets
    z_train, mag_train, color_train = train_datasets
    z_augmented, mag_augmented, color_augmented = augmented_datasets
    
    figure = pyplot.figure(figsize = (9, 12))
    gridspec = GridSpec(nrows=1, ncols=2, figure=figure, top=0.95, bottom=0.75, hspace=0.2, wspace=0.0)
    
    plot = figure.add_subplot(gridspec[0,:])
    
    plot.hist(z_test, bins=z_bin, linewidth=1.0, density=True, histtype='step', color='black', label=r'$\mathrm{test}$')
    
    plot.hist(z_train, bins=z_bin, linewidth=1.0, density=True, histtype='step', color='darkred', label=r'$\mathrm{train}$')
    
    plot.hist(z_input, bins=z_bin, linewidth=1.0, density=True, histtype='step', color='darkblue', label=r'$\mathrm{input}$')
    
    plot.hist(z_augmented, bins=z_bin, linewidth=1.0, density=True, histtype='step', color='darkorange', label=r'$\mathrm{augmented}$')
    
    plot.legend()
    plot.set_xlim(z_bin.min(), z_bin.max())
    
    plot.set_xlabel(r'$z$')
    plot.set_ylabel(r'$\mathcal{P}(z)$')
    
    gridspec = GridSpec(nrows=2, ncols=2, figure=figure, top=0.70, bottom=0.15, hspace=0.0, wspace=0.0)
    
    plot = figure.add_subplot(gridspec[0,0])
    
    plot.text(3.0, 15.0, r'$\mathrm{test}$')
    
    map_test, color_bin, mag_bin, mesh_test = plot.hist2d(color_test, mag_test, bins=[color_bin, mag_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot.set_ylabel(r'$i$')
    plot.set_xticklabels([])
    plot.get_yticklabels()[0].set_visible(False)
    
    plot.set_ylim(mag_bin.min(), mag_bin.max())
    plot.set_xlim(color_bin.min(), color_bin.max())
    
    plot = figure.add_subplot(gridspec[0,1])
    
    plot.text(3.0, 15.0, r'$\mathrm{augmented}$')
    
    map_augmented, color_bin, mag_bin, mesh_augmented = plot.hist2d(color_augmented, mag_augmented, bins=[color_bin, mag_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot.set_yticklabels([])
    plot.set_xticklabels([])
    plot.set_ylim(mag_bin.min(), mag_bin.max())
    plot.set_xlim(color_bin.min(), color_bin.max())
    
    plot = figure.add_subplot(gridspec[1,0])
    
    plot.text(3.0, 15.0, r'$\mathrm{input}$')
    
    map_input, color_bin, mag_bin, mesh_input = plot.hist2d(color_input, mag_input, bins=[color_bin, mag_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot.set_ylim(mag_bin.min(), mag_bin.max())
    plot.set_xlim(color_bin.min(), color_bin.max())
    
    plot.set_ylabel(r'$i$')
    plot.set_xlabel(r'$g - z$')
    
    plot = figure.add_subplot(gridspec[1,1])
    
    plot.text(3.0, 15.0, r'$\mathrm{train}$')
    
    map_train, color_bin, mag_bin, mesh_train = plot.hist2d(color_train, mag_train, bins=[color_bin, mag_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot.set_ylim(mag_bin.min(), mag_bin.max())
    plot.set_xlim(color_bin.min(), color_bin.max())
    
    plot.set_yticklabels([])
    plot.set_xlabel(r'$g - z$')
    
    color_bar = figure.colorbar(mesh_input, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    color_bar.set_label(r'$\mathrm{Counts}$')
    figure.subplots_adjust(bottom=0.15)
    return figure

def main(path, index):
    """
    Main function.
    
    Arguments:
        path (str): The path to the base folder.
        index (int): The index.
    
    Returns:
        float: The duration.
    """
    # Data store
    start = time.time()
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Path
    plot_path = os.path.join(path, 'PLOT/')
    data_path = os.path.join(path, 'DATA/')
    
    # Augment datasets
    augment_name = os.path.join(data_path, 'SAMPLE/AUGMENT_SAMPLE.hdf5')
    augment_data = data_store.read_file(key='augment_data', path=augment_name, handle_class=core.data.TableHandle)()
    
    z_augment = augment_data['photometry']['redshift']
    mag_augment = augment_data['photometry']['mag_i_lsst']
    color_augment = augment_data['photometry']['mag_g_lsst'] - augment_data['photometry']['mag_z_lsst']
    
    # Input datasets
    input_name = os.path.join(data_path, 'SAMPLE/INPUT_SAMPLE{}.hdf5'.format(index))
    input_data = data_store.read_file(key='input_data', path=input_name, handle_class=core.data.TableHandle)()
    
    z_input = input_data['photometry']['redshift']
    mag_input = input_data['photometry']['mag_i_lsst']
    color_input = input_data['photometry']['mag_g_lsst'] - input_data['photometry']['mag_z_lsst']
    
    # Test datasets
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)()
    
    z_test = test_data['photometry']['redshift']
    mag_test = test_data['photometry']['mag_i_lsst']
    color_test = test_data['photometry']['mag_g_lsst'] - test_data['photometry']['mag_z_lsst']
    
    # Bin Datasets
    z1 = 0.0
    z2 = 3.0
    z_bin_size = 150
    z_bin = numpy.linspace(z1, z2, z_bin_size + 1)
    
    mag1 = 12.0
    mag2 = 26.0
    mag_bin_size = 75
    mag_bin = numpy.linspace(mag1, mag2, mag_bin_size + 1)
    
    color1 = -1.0
    color2 = +6.0
    color_bin_size = 75
    color_bin = numpy.linspace(color1, color2, color_bin_size + 1)
    
    bin_datasets = [z_bin, mag_bin, color_bin]
    test_datasets = [z_test, mag_test, color_test]
    input_datasets = [z_input, mag_input, color_input]
    augment_datasets = [z_augment, mag_augment, color_augment]
    
    # Dataset Augmentation
    select_data = select(bin_datasets, input_datasets, augment_datasets)
    train_data = augment(input_data, augment_data, select_data)
    
    z_train = train_data['redshift']
    mag_train = train_data['mag_i_lsst']
    color_train = train_data['mag_g_lsst'] - train_data['mag_z_lsst']
    
    train_datasets = [z_train, mag_train, color_train]
    augmented_datasets = [z_augment[select_data], mag_augment[select_data], color_augment[select_data]]
    
    with h5py.File(os.path.join(data_path, 'SAMPLE/TRAIN_SAMPLE{}.hdf5'.format(index)), 'w') as file:
        group = file.create_group('photometry')
        for key, value in train_data.items():
            group.create_dataset(key, data=value)
    
    # Figure 
    figure = plot(bin_datasets, test_datasets, input_datasets, train_datasets, augmented_datasets)
    figure.savefig(plot_path + 'SAMPLE/SAMPLE{}.pdf'.format(index), bbox_inches = 'tight')
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Augmentation sample.')
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