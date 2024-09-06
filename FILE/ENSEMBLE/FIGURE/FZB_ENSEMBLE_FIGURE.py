import os
import time
import h5py
import numpy
import argparse
from matplotlib import pyplot


def main(path):
    # Data
    start = time.time()
    data_path = os.path.join(path, 'DATA/')
    plot_path = os.path.join(path, 'PLOT/')
    os.makedirs(os.path.join(plot_path, 'ENSEMBLE/'), exist_ok=True)
    
    # Ensemble
    with h5py.File(os.path.join(data_path, 'ENSEMBLE/LENS/FZB_ENSEMBLE_SELECT.hdf5'), 'r') as file:
        lens_data_select = file['data'][:].astype(numpy.float32)
        lens_sample_select = file['sample'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(data_path, 'ENSEMBLE/SOURCE/FZB_ENSEMBLE_SELECT.hdf5'), 'r') as file:
        source_data_select = file['data'][:].astype(numpy.float32)
        source_sample_select = file['sample'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(data_path, 'ENSEMBLE/LENS/FZB_ENSEMBLE.hdf5'), 'r') as file:
        lens_data = file['data'][:].astype(numpy.float32)
        lens_sample = file['sample'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(data_path, 'ENSEMBLE/SOURCE/FZB_ENSEMBLE.hdf5'), 'r') as file:
        source_data = file['data'][:].astype(numpy.float32)
        source_sample = file['sample'][:].astype(numpy.float32)
    
    z1 = 0.0
    z2 = 3.0
    size = 100
    bin_size = 5
    grid_size = 300
    z_delta = (z2 - z1) / grid_size
    z_data = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, grid_size)
    
    lens_mean_data_select = numpy.sum(lens_data_select * z_data[numpy.newaxis, :], axis=1) * z_delta
    lens_mean_sample_select = numpy.sum(lens_sample_select * z_data[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    
    source_mean_data_select = numpy.sum(source_data_select * z_data[numpy.newaxis, :], axis=1) * z_delta
    source_mean_sample_select = numpy.sum(source_sample_select * z_data[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    
    lens_mean_data = numpy.sum(lens_data * z_data[numpy.newaxis, :], axis=1) * z_delta
    lens_mean_sample = numpy.sum(lens_sample * z_data[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    
    source_mean_data = numpy.sum(source_data * z_data[numpy.newaxis, :], axis=1) * z_delta
    source_mean_sample = numpy.sum(source_sample * z_data[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot
    shift1 = -0.035
    shift2 = +0.035
    figure, plot = pyplot.subplots(ncols=2, nrows=bin_size, figsize=(12, 15))
    
    for m in range(bin_size):
        plot[m, 0].hist(lens_mean_sample_select[:, m], bins=size, range=(lens_mean_data_select[m] + shift1, lens_mean_data_select[m] + shift2), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(lens_mean_sample[:, m], bins=size, range=(lens_mean_data_select[m] + shift1, lens_mean_data_select[m] + shift2), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].text(lens_mean_data_select[m] + 0.5 * shift1, 250, r'$\Delta \langle z \rangle = {:.3f}$'.format(lens_mean_data[m] - lens_mean_data_select[m]), fontsize=20, ha='center')
        
        plot[m, 0].set_ylim(0, 350)
        plot[m, 0].set_yticklabels([])
        plot[m, 0].set_xlim(lens_mean_data_select[m] + shift1, lens_mean_data_select[m] + shift2)
        
        plot[m, 1].hist(source_mean_sample_select[:, m], bins=size, range=(source_mean_data_select[m] + shift1, source_mean_data_select[m] + shift2), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(source_mean_sample[:, m], bins=size, range=(source_mean_data_select[m] + shift1, source_mean_data_select[m] + shift2), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].text(source_mean_data_select[m] + 0.5 * shift1, 250, r'$\Delta \langle z \rangle = {:.3f}$'.format(source_mean_data[m] - source_mean_data_select[m]), fontsize=20, ha='center')
        
        plot[m, 1].set_ylim(0, 350)
        plot[m, 1].set_yticklabels([])
        plot[m, 1].set_xlim(source_mean_data_select[m] + shift1, source_mean_data_select[m] + shift2)
    
    figure.supxlabel(r'$\langle z \rangle$')
    figure.supylabel(r'$\mathcal{P} \left( \langle z \rangle \right)$')
    
    figure.subplots_adjust(wspace=0.0, hspace=0.2)
    figure.savefig(os.path.join(plot_path, 'ENSEMBLE/FZB_ENSEMBLE.pdf'), bbox_inches='tight')
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='Ensemble')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    
    PATH = PARSE.parse_args().path
    RESULT = main(PATH)