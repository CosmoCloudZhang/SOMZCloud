import os
import time
import h5py
import numpy
import argparse
from rail import core
import multiprocessing
from matplotlib import pyplot, colors

def plot_select(z_lens, z_mean, z_true, z_source, mag0_lens, mag0_source, mag_source):
    """
    Plot the magnitude distribution of source redshifts.
    
    Parameters:
        z_lens (numpy.ndarray): The redshift grid of lens samples.
        z_mean (numpy.ndarray): The redshift mode of source samples.
        z_true (numpy.ndarray): The redshifts of test application samples.
        z_source (numpy.ndarray): The redshift grid of source samples.
        mag0_lens (float): The magnitude threshold of lens samples.
        mag0_source (float): The magnitude threshold of source samples.
        mag_source (numpy.ndarray): The magnitudes of test application samples.
    
    Returns:
        matplotlib.figure.Figure: The plotted figure.
    
    """
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Set variables
    z1_lens = z_lens.min()
    z2_lens = z_lens.max()
    
    z1_source = z_source.min()
    z2_source = z_source.max()
    
    z_bin_size = 100
    z_bin = numpy.linspace(z1_source, z2_source, z_bin_size + 1)
    
    mag1 = 16.0
    mag2 = 26.0
    mag_bin_size = 100
    mag_bin = numpy.linspace(mag1, mag2, mag_bin_size + 1)
    
    # Plot
    width = 10
    figure, plot = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 8))
    
    # Plot 1
    z_mesh = plot[0].hist2d(x=z_mean, y=mag_source, bins=[z_bin, mag_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')[-1]
    
    plot[0].plot(z_source, numpy.ones_like(z_source) * mag0_source, color='black', linestyle='--', linewidth=2.0)
    
    plot[0].plot(z_source[(z1_lens <= z_source) & (z_source <= 1.5)], 4 * z_source[(z1_lens <= z_source) & (z_source <= 1.5)] + 18, color='black', linestyle='-', linewidth=2.0)
    
    plot[0].plot(z_source[(1.5 <= z_source) & (z_source <= z2_lens)], numpy.ones_like(z_source[(1.5 <= z_source) & (z_source <= z2_lens)]) * mag0_lens, color='black', linestyle='-', linewidth=2.0)
    
    plot[0].plot(numpy.ones(width + 1) * z2_lens, numpy.linspace(mag1, mag0_lens, width + 1), color='black', linestyle='-', linewidth=2.0)
    
    plot[0].set_xlim(z1_source, z2_source)
    plot[0].set_ylim(mag1, mag2)
    
    plot[0].set_ylabel(r'$i$')
    plot[0].set_xlabel(r'$z_\mathrm{mode}$')
    
    # Plot 2
    z_mesh = plot[1].hist2d(x=z_true, y=mag_source, bins=[z_bin, mag_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')[-1]
    
    plot[1].plot(z_source, numpy.ones_like(z_source) * mag0_source, color='black', linestyle='--', linewidth=2.0)
    
    plot[1].plot(z_source[(z1_lens <= z_source) & (z_source <= 1.5)], 4 * z_source[(z1_lens <= z_source) & (z_source <= 1.5)] + 18, color='black', linestyle='-', linewidth=2.0)
    
    plot[1].plot(z_source[(1.5 <= z_source) & (z_source <= z2_lens)], numpy.ones_like(z_source[(1.5 <= z_source) & (z_source <= z2_lens)]) * mag0_lens, color='black', linestyle='-', linewidth=2.0)
    
    plot[1].plot(numpy.ones(width + 1) * z2_lens, numpy.linspace(mag1, mag0_lens, width + 1), color='black', linestyle='-', linewidth=2.0)
    
    plot[1].set_xlim(z1_source, z2_source)
    plot[1].set_ylim(mag1, mag2)
    
    plot[1].set_xlabel(r'$z_\mathrm{true}$')
    
    plot[1].set_yticklabels([])
    plot[1].get_xticklabels()[0].set_visible(False)
    
    # Color bar
    color_bar = figure.colorbar(z_mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    color_bar.set_label(r'$\mathrm{Counts}$')
    
    # Return figure
    figure.subplots_adjust(bottom=0.20, wspace=0.00)
    return figure


def plot_redshift(z_lens, z_mean, z_true, z_source, mag0_lens, mag0_source, mag_source):
    """
    Plot the photometric redshift distribution of source samples.
    
    Parameters:
        z_lens (numpy.ndarray): The redshift grid of lens samples.
        z_mean (numpy.ndarray): The redshift mode of source samples.
        z_true (numpy.ndarray): The redshifts of test application samples.
        z_source (numpy.ndarray): The redshift grid of source samples.
        mag0_lens (float): The magnitude threshold of lens samples.
        mag0_source (float): The magnitude threshold of source samples.
        mag_source (numpy.ndarray): The magnitudes of test application samples.
    
    Returns:
        matplotlib.figure.Figure: The plotted figure.
    
    """
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Set variables
    z1_lens = z_lens.min()
    z2_lens = z_lens.max()
    
    z1_source = z_source.min()
    z2_source = z_source.max()
    
    z_bin_size = 100
    z_bin = numpy.linspace(z1_source, z2_source, z_bin_size + 1)
    
    select_source = (z1_source < z_mean) & (z_mean < z2_source) & (mag_source < mag0_source)
    select_lens = (z1_lens < z_mean) & (z_mean < z2_lens) & (mag_source < 4 * z_mean + 18) & (mag_source < mag0_lens)
    
    # Plot
    figure, plot = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 8))
    
    # Plot 1
    z_mesh = plot[0].hist2d(x=z_true[select_lens], y=z_mean[select_lens], bins=[z_bin, z_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')[-1]
    
    plot[0].plot(z_source, z_source, color='black', linestyle='--', linewidth=2.0)
    
    plot[0].set_xlim(z1_source, z2_source)
    plot[0].set_ylim(z1_source, z2_source)
    
    plot[0].set_title(r'$\mathrm{Lens}$')
    plot[0].set_ylabel(r'$z_\mathrm{mode}$')
    plot[0].set_xlabel(r'$z_\mathrm{true}$')
    
    # Plot 2
    z_mesh = plot[1].hist2d(x=z_true[select_source], y=z_mean[select_source], bins=[z_bin, z_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')[-1]
    
    plot[1].plot(z_source, z_source, color='black', linestyle='--', linewidth=2.0)
    
    plot[1].set_xlim(z1_source, z2_source)
    plot[1].set_ylim(z1_source, z2_source)
    
    plot[1].set_title(r'$\mathrm{Source}$')
    plot[1].set_xlabel(r'$z_\mathrm{true}$')
    
    plot[1].set_yticklabels([])
    plot[1].get_xticklabels()[0].set_visible(False)
    
    # Color bar
    color_bar = figure.colorbar(z_mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    color_bar.set_label(r'$\mathrm{Counts}$')
    
    # Return figure
    figure.subplots_adjust(bottom=0.20, wspace=0.00)
    return figure


def save_pdf(z_lens, z_pdf, z_mean, z_source, bin_lens, bin_source, mag0_lens, mag0_source, mag_source):
    """
    Save the selected samples.
    
    Parameters:
        z_lens (numpy.ndarray): The redshift grid of lens samples.
        z_pdf (numpy.ndarray): The redshift PDF of source samples.
        z_mean (numpy.ndarray): The redshift mode of source samples.
        z_source (numpy.ndarray): The redshift grid of source samples.
        bin_lens (numpy.ndarray): The redshift bin of lens samples.
        bin_source (numpy.ndarray): The redshift bin of source samples.
        mag0_lens (float): The magnitude threshold of lens samples.
        mag0_source (float): The magnitude threshold of source samples.
        mag_source (numpy.ndarray): The magnitudes of test application samples.
    
    Returns:
        tuple: The selected lens and source samples.
    """
    # Select
    z1_lens = z_lens.min()
    z2_lens = z_lens.max()
    
    z1_source = z_source.min()
    z2_source = z_source.max()
    
    select_source = (z1_source < z_mean) & (z_mean < z2_source) & (mag_source < mag0_source)
    select_lens = (z1_lens < z_mean) & (z_mean < z2_lens) & (mag_source < 4 * z_mean + 18) & (mag_source < mag0_lens)
    meta = {'pdf_name': numpy.array(['interp'.encode('ascii')]).astype('S6'), 'pdf_version': numpy.array([0]).astype(numpy.int32), 'xvals': numpy.array([z_source]).astype(numpy.float32)}
    
    # Lens
    lens = []
    lens_size = len(bin_lens) - 1
    for m in range(lens_size):
        select = select_lens & (bin_lens[m] < z_mean) & (z_mean < bin_lens[m + 1])
        data = {'yvals': z_pdf[select, :].astype(numpy.float32)}
        lens.append({'data': data, 'meta': meta})
    
    # Source
    source = []
    source_size = len(bin_source) - 1
    for n in range(source_size):
        select = select_source & (bin_source[n] < z_mean) & (z_mean < bin_source[n + 1])
        data = {'yvals': z_pdf[select, :].astype(numpy.float32)}
        source.append({'data': data, 'meta': meta})
    
    # Return
    return lens, source


def save_data(z_lens, z_mean, z_true, z_source, bin_lens, bin_source, mag0_lens, mag0_source, mag_source):
    """
    Save the selected samples.
    
    Parameters:
        z_lens (numpy.ndarray): The redshift grid of lens samples.
        z_mean (numpy.ndarray): The redshift mode of source samples.
        z_true (numpy.ndarray): The redshifts of test application samples.
        z_source (numpy.ndarray): The redshift grid of source samples.
        bin_lens (numpy.ndarray): The redshift bin of lens samples.
        bin_source (numpy.ndarray): The redshift bin of source samples.
        mag0_lens (float): The magnitude threshold of lens samples.
        mag0_source (float): The magnitude threshold of source samples.
        mag_source (numpy.ndarray): The magnitudes of test application samples.
    
    Returns:
        tuple: The selected lens and source true redshift distributions.
    """
    # Select
    z1_lens = z_lens.min()
    z2_lens = z_lens.max()
    
    z1_source = z_source.min()
    z2_source = z_source.max()
    
    select_source = (z1_source < z_mean) & (z_mean < z2_source) & (mag_source < mag0_source)
    select_lens = (z1_lens < z_mean) & (z_mean < z2_lens) & (mag_source < 4 * z_mean + 18) & (mag_source < mag0_lens)
    
    # Lens
    grid_size = z_source.size - 1
    lens_size = len(bin_lens) - 1
    lens_data = numpy.zeros((lens_size, grid_size), dtype=numpy.float32)
    
    for m in range(lens_size):
        select = select_lens & (bin_lens[m] < z_mean) & (z_mean < bin_lens[m + 1])
        lens_data[m, :] = numpy.histogram(z_true[select], bins=z_source, range=(z1_source, z2_source), density=False)[0].astype(numpy.float32)
    
    lens_count = numpy.sum(lens_data, axis=1)
    lens = {'data': lens_data, 'count': lens_count}
    
    # Source
    grid_size = z_source.size - 1
    source_size = len(bin_source) - 1
    source_data = numpy.zeros((source_size, grid_size), dtype=numpy.float32)
    
    for n in range(source_size):
        select = select_source & (bin_source[n] < z_mean) & (z_mean < bin_source[n + 1])
        source_data[n, :] = numpy.histogram(z_true[select], bins=z_source, range=(z1_source, z2_source), density=False)[0].astype(numpy.float32)
    
    source_count = numpy.sum(source_data, axis=1)
    source = {'data': source_data, 'count': source_count}
    
    # Return
    return lens, source


def main(path, index):
    start = time.time()
    
    # Data store
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Data
    data_path = os.path.join(path, 'DATA/')
    plot_path = os.path.join(path, 'PLOT/')
    
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    estimate_name = os.path.join(data_path, 'ESTIMATE/FZB_ESTIMATE{}.hdf5'.format(index))
    
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)
    
    # Bin
    bin_name = os.path.join(data_path, 'BIN/BIN.hdf5')
    with h5py.File(bin_name, 'r') as file:
        bin_lens = file['lens'][:].astype(numpy.float32)
        bin_source = file['source'][:].astype(numpy.float32)
    
    # Redshift
    z1_lens = 0.0
    z2_lens = 2.0
    
    z1_source = 0.0
    z2_source = 3.0
    
    z_lens_size = 200
    z_lens = numpy.linspace(z1_lens, z2_lens, z_lens_size + 1)
    
    z_source_size = 300
    z_source = numpy.linspace(z1_source, z2_source, z_source_size + 1)
    
    z_true = test_data()['photometry']['redshift']
    mag_source = test_data()['photometry']['mag_i_lsst']
    
    z_pdf = estimator().pdf(z_source)
    z_mean = numpy.concatenate(estimator().mean())
    
    # Magnitude
    mag0_lens = 24.0
    mag1_source = 25.0
    mag2_source = 25.2
    mag0_source =  numpy.random.uniform(mag1_source, mag2_source)
    
    # Plot
    figure = plot_select(z_lens, z_mean, z_true, z_source, mag0_lens, mag0_source, mag_source)
    figure.savefig(os.path.join(plot_path, 'SELECT/SELECT{}.pdf'.format(index)), bbox_inches='tight')
    pyplot.close(figure)
    
    figure = plot_redshift(z_lens, z_mean, z_true, z_source, mag0_lens, mag0_source, mag_source)
    figure.savefig(os.path.join(plot_path, 'REDSHIFT/REDSHIFT{}.pdf'.format(index)), bbox_inches='tight')
    pyplot.close(figure)
    
    # Save PDF
    lens_pdf, source_pdf = save_pdf(z_lens, z_pdf, z_mean, z_source, bin_lens, bin_source, mag0_lens, mag0_source, mag_source)
    
    # Lens
    lens_size = len(bin_lens) - 1
    os.makedirs(os.path.join(data_path, 'LENS/LENS{}'.format(index)), exist_ok=True)
    
    for m in range(lens_size):        
        with h5py.File(os.path.join(data_path, 'LENS/LENS{}/SELECT{}.hdf5'.format(index, m)), 'w') as file:
            for name in lens_pdf[m].keys():
                file.create_group(name)
                for key, value in lens_pdf[m][name].items():
                    file[name].create_dataset(key, data=value)
    
    # Source
    source_size = len(bin_source) - 1
    os.makedirs(os.path.join(data_path, 'SOURCE/SOURCE{}'.format(index)), exist_ok=True)
    
    for n in range(source_size):
        with h5py.File(os.path.join(data_path, 'SOURCE/SOURCE{}/SELECT{}.hdf5'.format(index, n)), 'w') as file:
            for name in source_pdf[n].keys():
                file.create_group(name)
                for key, value in source_pdf[n][name].items():
                    file[name].create_dataset(key, data=value)
    
    # Save Data
    lens_data, source_data = save_data(z_lens, z_mean, z_true, z_source, bin_lens, bin_source, mag0_lens, mag0_source, mag_source)
    
    with h5py.File(os.path.join(data_path, 'LENS/LENS{}/SELECT.hdf5'.format(index)), 'w') as file:
        for key, value in lens_data.items():
            file.create_dataset(key, data=value)
    
    with h5py.File(os.path.join(data_path, 'SOURCE/SOURCE{}/SELECT.hdf5'.format(index)), 'w') as file:
        for key, value in source_data.items():
            file.create_dataset(key, data=value)
    
    # Return
    end = time.time()
    print('Index:{}, Time: {:.2f} minutes'.format(index, (end - start) / 60))
    return index

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Informer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the processes')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets')
    
    PATH = PARSE.parse_args().path
    NUMBER = PARSE.parse_args().number
    LENGTH = PARSE.parse_args().length
    
    # Multiprocessing
    with multiprocessing.Pool(processes=NUMBER) as POOL:
        POOL.starmap(main, [(PATH, index) for index in range(1, LENGTH + 1)])