import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot


def plot_ensemble(z_grid, select_lens, select_source, bin_lens_size, bin_source_size):
    '''
    Plot the ensemble of the lens and source redshift distributions
    
    Arguments:
        z_grid (numpy.ndarray): The redshift grid
        select_lens (numpy.ndarray): The lens distributions
        select_source (numpy.ndarray): The source distributions
        bin_lens_size (int): The number of lens tomographic bins
        bin_source_size (int): The number of source tomographic bins
    
    Returns:
        figure (matplotlib.figure.Figure): The figure
    '''
    # Figure
    figure, plot = pyplot.subplots(nrows = 2, ncols = 1, figsize = (12, 12))
    color_lens_list = ['darkmagenta', 'darkblue', 'darkgreen', 'darkorange', 'darkred']
    color_source_list = ['darkmagenta', 'darkblue', 'darkgreen', 'darkorange', 'darkred']
    
    for m in range(bin_lens_size):
        
        plot[0].plot(z_grid, numpy.transpose(select_lens[:, m, :]), color = color_lens_list[m], linewidth = 0.1, alpha = 0.1, rasterized=True)
        
        plot[0].plot(z_grid, numpy.mean(select_lens[:, m, :], axis=0), color = color_lens_list[m], linewidth = 2.5, label=r'$\mathrm{Bin} \,' + r'{:.0f}$'.format(m + 1))
    
    plot[0].set_xlim(0.0, 2.0)
    plot[0].set_ylim(0.0, 8.0)
    
    plot[0].set_xticklabels([])
    plot[0].set_yticks([2.0, 4.0, 6.0, 8.0])
    plot[0].legend(loc='upper right', fontsize=20)
    
    plot[0].set_ylabel(r'$\phi \left( z \right)$')
    plot[0].text(x=1.6, y=3.0, s=r'$\mathrm{Lens}$', fontsize=25)
    
    for m in range(bin_source_size):
        
        plot[1].plot(z_grid, numpy.transpose(select_source[:, m, :]), color = color_source_list[m], linewidth = 0.1, alpha = 0.1, rasterized=True)
        
        plot[1].plot(z_grid, numpy.mean(select_source[:, m, :], axis=0), color = color_source_list[m], linewidth = 2.5, label=r'$\mathrm{Bin} \,' + r'{:.0f}$'.format(m + 1))
    
    plot[1].set_xlim(0.0, 2.0)
    plot[1].set_ylim(0.0, 8.0)
    
    plot[1].set_yticks([0.0, 2.0, 4.0, 6.0, 8.0])
    plot[1].legend(loc='upper right', fontsize=20)
    
    plot[1].set_xlabel(r'$z$')
    plot[1].set_ylabel(r'$\phi \left( z \right)$')
    plot[1].text(x=1.6, y=3.0, s=r'$\mathtt{Source}$', fontsize=25)
    
    figure.subplots_adjust(hspace=0.0)
    return figure


def main(tag, name, label, folder):
    '''
    Plot the ensemble of the lens and source redshift distributions
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    print('Name: {}, Label: {}'.format(name, label))
    
    # Path
    prior_folder = os.path.join(folder, 'PRIOR/')
    calibrate_folder = os.path.join(folder, 'CALIBRATE/')
    os.makedirs(os.path.join(prior_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(prior_folder, '{}/ENSEMBLE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(prior_folder, '{}/ENSEMBLE/{}/'.format(tag, name)), exist_ok=True)
    os.makedirs(os.path.join(prior_folder, '{}/ENSEMBLE/{}/{}/'.format(tag, name, label)), exist_ok=True)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 30
    
    # Shift
    with h5py.File(os.path.join(calibrate_folder, '{}/SHIFT/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta'].keys()}
        
        data_lens = file['lens']['data'][...]
        data_source = file['source']['data'][...]
    
    # Size
    select_size = 10000
    data_size, bin_lens_size, z_size = data_lens.shape
    data_size, bin_source_size, z_size = data_source.shape
    indices = numpy.sort(numpy.random.choice(data_size, select_size, replace=False))
    
    # Select
    z1 = meta['z1']
    z2 = meta['z2']
    z_grid = numpy.linspace(z1, z2, z_size)
    
    select_lens = data_lens[indices, :, :]
    select_source = data_source[indices, :, :]
    
    # Plot
    figure = plot_ensemble(z_grid, select_lens, select_source, bin_lens_size, bin_source_size)
    figure.savefig(os.path.join(prior_folder, '{}/ENSEMBLE/{}/{}/SHIFT.pdf'.format(tag, name, label)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Scale
    with h5py.File(os.path.join(calibrate_folder, '{}/SCALE/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta'].keys()}
        
        scale_lens = file['lens']['data'][...]
        scale_source = file['source']['data'][...]
    
    data_size, bin_lens_size, z_size = scale_lens.shape
    data_size, bin_source_size, z_size = scale_source.shape
    
    select_scale_lens = scale_lens[indices, :, :]
    select_scale_source = scale_source[indices, :, :]
    
    # Plot
    figure = plot_ensemble(z_grid, select_scale_lens, select_scale_source, bin_lens_size, bin_source_size)
    figure.savefig(os.path.join(prior_folder, '{}/ENSEMBLE/{}/{}/SCALE.pdf'.format(tag, name, label)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Correct
    with h5py.File(os.path.join(calibrate_folder, '{}/CORRECT/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta'].keys()}
        
        data_lens = file['lens']['data'][...]
        data_source = file['source']['data'][...]
    
    data_size, bin_lens_size, z_size = data_lens.shape
    data_size, bin_source_size, z_size = data_source.shape
    
    select_correct_lens = data_lens[indices, :, :]
    select_correct_source = data_source[indices, :, :]
    
    # Plot
    figure = plot_ensemble(z_grid, select_correct_lens, select_correct_source, bin_lens_size, bin_source_size)
    figure.savefig(os.path.join(prior_folder, '{}/ENSEMBLE/{}/{}/CORRECT.pdf'.format(tag, name, label)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Analysis Ensemble')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    name = PARSE.parse_args().name
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, name, LABEL, FOLDER)