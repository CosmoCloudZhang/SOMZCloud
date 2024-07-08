import os
import h5py
import time
import numpy
import argparse
from rail import core
import multiprocessing
from matplotlib import pyplot
from matplotlib import colors
from matplotlib.gridspec import GridSpec

def bound(z, mag, color):
    
    # Magnitude
    mag_quantile = 0.75
    mag0 = numpy.quantile(mag, mag_quantile)
    
    # Color
    color_quantile = 0.25
    color0 = numpy.quantile(color, color_quantile)
    
    # Count
    count0 = len(z) // 2
    return mag0, color0, count0

def select(z, mag, color, z0, mag0, color0, count0):
    
    z1 = 0.0
    z2 = 3.0
    z_bin_size = 300
    z_delta = (z2 - z1) / z_bin_size
    z_edge = numpy.linspace(z1, z2, z_bin_size + 1)
    z_bin = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, z_bin_size)
    
    pdf = numpy.histogram(z0, bins=z_edge, density=True)[0]
    distribution = numpy.log(1 + numpy.exp(-pdf))
    
    mask1 = (color < color0)
    mask2 = (color > color0) & (mag > mag0)
    
    mask = (mask1 | mask2)
    mask_indices = numpy.where(mask)[0]
    
    probability = numpy.interp(z[mask_indices], z_bin, distribution)
    probability = probability / numpy.sum(probability)
    size0 = numpy.minimum(len(mask_indices), count0)
    
    mask_sample = numpy.random.choice(mask_indices, size=size0, replace=False, p=probability)
    select_sample = numpy.zeros_like(mask, dtype=bool)
    select_sample[mask_sample] = True
    
    return select_sample

def augment(data_store, augment_data, input_data, select_data):
    
    train_table = {}
    for key in input_data().keys():
        train_table[key] = numpy.concatenate([input_data()[key], augment_data()[key][select_data]])
    
    train_data = data_store.add_data('train_data', train_table, core.data.TableHandle)
    return train_data

def plot(bin_datasets, test_datasets, train_datasets, input_datasets, augment_datasets):
    
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 15
    
    z_bin, mag_bin, color_bin = bin_datasets
    z_test, mag_test, color_test = test_datasets
    z_train, mag_train, color_train = train_datasets
    z_input, mag_input, color_input = input_datasets
    z_augment, mag_augment, color_augment = augment_datasets
    mag0_input, color0_input, count0_input = bound(z_input, mag_input, color_input)
    
    figure = pyplot.figure(figsize = (9, 12))
    gridspec = GridSpec(nrows=1, ncols=2, figure=figure, top=0.95, bottom=0.75, hspace=0.2, wspace=0.0)
    
    plot = figure.add_subplot(gridspec[0,:])
    
    plot.hist(z_test, bins=z_bin, linewidth=2.0, density=True, histtype='step', color='black', label=r'$\mathrm{test}$')
    
    plot.hist(z_train, bins=z_bin, linewidth=2.0, density=True, histtype='step', color='darkred', label=r'$\mathrm{train}$')
    
    plot.hist(z_input, bins=z_bin, linewidth=2.0, density=True, histtype='step', color='darkblue', label=r'$\mathrm{input}$')
    
    plot.hist(z_augment, bins=z_bin, linewidth=2.0, density=True, histtype='step', color='darkgreen', label=r'$\mathrm{augment}$')
    
    plot.legend()
    plot.set_xlim(z_bin.min(), z_bin.max())
    
    plot.set_xlabel(r'$z$')
    plot.set_ylabel(r'$\mathcal{P}(z)$')
    
    gridspec = GridSpec(nrows=2, ncols=2, figure=figure, top=0.70, bottom=0.15, hspace=0.0, wspace=0.0)
    
    plot = figure.add_subplot(gridspec[0,0])
    
    plot.text(3.0, 15.0, r'$\mathrm{augment}$')
    
    map_augment, color_bin, mag_bin, mesh_augment = plot.hist2d(color_augment, mag_augment, bins=[color_bin, mag_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot.set_ylabel(r'$i$')
    plot.set_xticklabels([])
    plot.set_ylim(mag_bin.min(), mag_bin.max())
    plot.set_xlim(color_bin.min(), color_bin.max())
    
    plot = figure.add_subplot(gridspec[0,1])
    
    plot.text(3.0, 15.0, r'$\mathrm{test}$')
    
    map_test, color_bin, mag_bin, mesh_test = plot.hist2d(color_test, mag_test, bins=[color_bin, mag_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot.set_yticklabels([])
    plot.set_xticklabels([])
    plot.set_ylim(mag_bin.min(), mag_bin.max())
    plot.set_xlim(color_bin.min(), color_bin.max())
    
    plot = figure.add_subplot(gridspec[1,0])
    
    plot.plot(numpy.linspace(color0_input, color_bin.max(), 10), numpy.ones(10) * mag0_input, color='black', linestyle='--', linewidth=2.0)
    
    plot.plot(numpy.ones(10) * color0_input, numpy.linspace(mag_bin.min(), mag0_input, 10), color='black', linestyle='--', linewidth=2.0)
    
    plot.annotate('', xy=(color0_input - 1.0, 13.5), xytext=(color0_input, 13.5), arrowprops=dict(arrowstyle='->', color='black', linewidth=2.0))
    
    plot.annotate('', xy=(4.5, mag0_input + 2.0), xytext=(4.5, mag0_input), arrowprops=dict(arrowstyle='->', color='black', linewidth=2.0))
    
    plot.text(3.0, 15.0, r'$\mathrm{input}$')
    
    map_input, color_bin, mag_bin, mesh_input = plot.hist2d(color_input, mag_input, bins=[color_bin, mag_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot.set_ylim(mag_bin.min(), mag_bin.max())
    plot.set_xlim(color_bin.min(), color_bin.max())
    
    plot.set_ylabel(r'$i$')
    plot.set_xlabel(r'$g - z$')
    
    plot = figure.add_subplot(gridspec[1,1])
    
    mag0_input, color0_input, count0_input = bound(z_input, mag_input, color_input)
    
    plot.plot(numpy.linspace(color0_input, color_bin.max(), 10), numpy.ones(10) * mag0_input, color='black', linestyle='--', linewidth=2.0)
    
    plot.plot(numpy.ones(10) * color0_input, numpy.linspace(mag_bin.min(), mag0_input, 10), color='black', linestyle='--', linewidth=2.0)
    
    plot.annotate('', xy=(color0_input - 1.0, 13.5), xytext=(color0_input, 13.5), arrowprops=dict(arrowstyle='->', color='black', linewidth=2.0))
    
    plot.annotate('', xy=(4.5, mag0_input + 2.0), xytext=(4.5, mag0_input), arrowprops=dict(arrowstyle='->', color='black', linewidth=2.0))
    
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
    start = time.time()
    
    # Data store
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Path
    plot_path = os.path.join(path, 'PLOT/')
    data_path = os.path.join(path, 'DATA/')
    
    # Augment datasets
    augment_name = os.path.join(data_path, 'SAMPLE/AUGMENT_SAMPLE.hdf5')
    augment_file = h5py.File(augment_name, 'r')
    
    augment_table = {}
    augment_group = augment_file['photometry']
    for key in augment_group.keys():
        augment_table[key] = augment_group[key][:].astype('float32')
    
    augment_file.close()
    augment_data = data_store.add_data('augment_data', augment_table, core.data.TableHandle)
    
    z_augment = augment_data()['redshift']
    mag_augment = augment_data()['mag_i_lsst']
    color_augment = augment_data()['mag_g_lsst'] - augment_data()['mag_z_lsst']
    
    # Input datasets
    input_name = os.path.join(data_path, 'SAMPLE/INPUT_SAMPLE{}.hdf5'.format(index))
    input_file = h5py.File(input_name, 'r')
    
    input_table = {}
    input_group = input_file['photometry']
    for key in input_group.keys():
        input_table[key] = input_group[key][:].astype('float32')
    
    input_file.close()
    input_data = data_store.add_data('input_data', input_table, core.data.TableHandle)
    
    z_input = input_data()['redshift']
    mag_input = input_data()['mag_i_lsst']
    color_input = input_data()['mag_g_lsst'] - input_data()['mag_z_lsst']
    
    # Test datasets
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    test_file = h5py.File(test_name, 'r')
    
    test_table = {}
    test_group = test_file['photometry']
    for key in test_group.keys():
        test_table[key] = test_group[key][:].astype('float32')
    
    test_file.close()
    test_data = data_store.add_data('test_data', test_table, core.data.TableHandle)
    
    z_test = test_data()['redshift']
    mag_test = test_data()['mag_i_lsst']
    color_test = test_data()['mag_g_lsst'] - test_data()['mag_z_lsst']
    
    # Dataset Augmentation
    mag0_input, color0_input, count0_input = bound(z_input, mag_input, color_input)
    
    select_data = select(z_augment, mag_augment, color_augment, z_input, mag0_input, color0_input, count0_input)
    
    train_data = augment(data_store, augment_data, input_data, select_data)
    
    z_train = train_data()['redshift']
    mag_train = train_data()['mag_i_lsst']
    color_train = train_data()['mag_g_lsst'] - train_data()['mag_z_lsst']
    
    with h5py.File(os.path.join(data_path, 'SAMPLE/TRAIN_SAMPLE{}.hdf5'.format(index)), 'w') as file:
        group = file.create_group('photometry')
        for key, value in train_data().items():
            group.create_dataset(key, data=value)
    
    # Figure 
    z1 = 0.0
    z2 = 3.0
    z_bin_size = 100
    z_bin = numpy.linspace(z1, z2, z_bin_size + 1)
    
    mag1 = 11.5
    mag2 = 26.0
    mag_bin_size = 50
    mag_bin = numpy.linspace(mag1, mag2, mag_bin_size + 1)
    
    color1 = -1.0
    color2 = +6.0
    color_bin_size = 50
    color_bin = numpy.linspace(color1, color2, color_bin_size + 1)
    
    bin_datasets = [z_bin, mag_bin, color_bin]
    test_datasets = [z_test, mag_test, color_test]
    train_datasets = [z_train, mag_train, color_train]
    input_datasets = [z_input, mag_input, color_input]
    augment_datasets = [z_augment, mag_augment, color_augment]
    
    figure = plot(bin_datasets, test_datasets, train_datasets, input_datasets, augment_datasets)
    figure.savefig(plot_path + 'SAMPLE/SAMPLE{}.pdf'.format(index), bbox_inches = 'tight')
    pyplot.close(figure)
    
    # Return
    end = time.time()
    print('Index:{}, Time: {:.2f} minutes'.format(index, (end - start) / 60))
    return index


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