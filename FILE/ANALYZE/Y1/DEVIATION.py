import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot


def main(tag, label, folder):
    '''
    Plot the standard deviation of the lens and source redshift distributions
    
    Arguments:
        tag (str): The tag of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    print('Label: {}'.format(label))
    
    # Path
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    os.makedirs(os.path.join(analyze_folder, '{}/DEVIATION/'.format(tag)), exist_ok=True)
    
    # Info
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/SOM_{}.hdf5'.format(tag, label)), 'r') as file:
        som_scatter_lens = file['lens']['scatter'][...]
        som_scatter_source = file['source']['scatter'][...]
        
        som_deviation_lens = file['lens']['deviation'][...]
        som_deviation_source = file['source']['deviation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/MODEL_{}.hdf5'.format(tag, label)), 'r') as file:
        model_scatter_lens = file['lens']['scatter'][...]
        model_scatter_source = file['source']['scatter'][...]
        
        model_deviation_lens = file['lens']['deviation'][...]
        model_deviation_source = file['source']['deviation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/PRODUCT_{}.hdf5'.format(tag, label)), 'r') as file:
        product_scatter_lens = file['lens']['scatter'][...]
        product_scatter_source = file['source']['scatter'][...]
        
        product_deviation_lens = file['lens']['deviation'][...]
        product_deviation_source = file['source']['deviation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/FIDUCIAL_{}.hdf5'.format(tag, label)), 'r') as file:
        fiducial_scatter_lens = file['lens']['scatter'][...]
        fiducial_scatter_source = file['source']['scatter'][...]
        
        fiducial_deviation_lens = file['lens']['deviation'][...]
        fiducial_deviation_source = file['source']['deviation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/HISTOGRAM_{}.hdf5'.format(tag, label)), 'r') as file:
        histogram_scatter_lens = file['lens']['scatter'][...]
        histogram_scatter_source = file['source']['scatter'][...]
        
        histogram_deviation_lens = file['lens']['deviation'][...]
        histogram_deviation_source = file['source']['deviation'][...]
        
        histogram_middle_lens = file['lens']['middle'][...]
        histogram_middle_source = file['source']['middle'][...]
    
    # Delta
    som_delta_lens = numpy.abs(som_scatter_lens - histogram_scatter_lens) / (1 + histogram_middle_lens)
    som_delta_source = numpy.abs(som_scatter_source - histogram_scatter_source) / (1 + histogram_middle_source)
    
    model_delta_lens = numpy.abs(model_scatter_lens - histogram_scatter_lens) / (1 + histogram_middle_lens)
    model_delta_source = numpy.abs(model_scatter_source - histogram_scatter_source) / (1 + histogram_middle_source)
    
    product_delta_lens = numpy.abs(product_scatter_lens - histogram_scatter_lens) / (1 + histogram_middle_lens)
    product_delta_source = numpy.abs(product_scatter_source - histogram_scatter_source) / (1 + histogram_middle_source)
    
    fiducial_delta_lens = numpy.abs(fiducial_scatter_lens - histogram_scatter_lens) / (1 + histogram_middle_lens)
    fiducial_delta_source = numpy.abs(fiducial_scatter_source - histogram_scatter_source) / (1 + histogram_middle_source)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Variable
    size = 100
    bin_size = 5
    
    range_lens = 1.0 * histogram_scatter_lens
    range_source = 1.0 * histogram_scatter_source
    
    factor_lens = 0.005 * (1 + histogram_middle_lens)
    factor_source = 0.002 * (1 + histogram_middle_source)
    
    # Plot
    figure, plot = pyplot.subplots(nrows=bin_size, ncols=2, figsize=(12, 3 * bin_size))
    
    for m in range(bin_size):
        
        plot[m, 0].hist(som_deviation_lens[:, m], bins=size, range=(histogram_scatter_lens[m] - range_lens[m], histogram_scatter_lens[m] + range_lens[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(model_deviation_lens[:, m], bins=size, range=(histogram_scatter_lens[m] - range_lens[m], histogram_scatter_lens[m] + range_lens[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(product_deviation_lens[:, m], bins=size, range=(histogram_scatter_lens[m] - range_lens[m], histogram_scatter_lens[m] + range_lens[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(fiducial_deviation_lens[:, m], bins=size, range=(histogram_scatter_lens[m] - range_lens[m], histogram_scatter_lens[m] + range_lens[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(histogram_deviation_lens[:, m], bins=size, range=(histogram_scatter_lens[m] - range_lens[m], histogram_scatter_lens[m] + range_lens[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].text(x=histogram_scatter_lens[m] - range_lens[m] * 0.80, y=250, s=r'$\delta^\mathrm{SOM}_{\bar{\zeta}} = ' + r'{:.3f}$'.format(som_delta_lens[m]), fontsize=15, color='darkblue')
        
        plot[m, 0].text(x=histogram_scatter_lens[m] - range_lens[m] * 0.80, y=100, s=r'$\delta^\mathrm{Model}_{\bar{\zeta}} = ' + r'{:.3f}$'.format(model_delta_lens[m]), fontsize=15, color='darkgreen')
        
        plot[m, 0].text(x=histogram_scatter_lens[m] + range_lens[m] * 0.24, y=250, s=r'$\delta^\mathrm{Product}_{\bar{\zeta}} = ' + r'{:.3f}$'.format(product_delta_lens[m]), fontsize=15, color='darkorange')
        
        plot[m, 0].text(x=histogram_scatter_lens[m] + range_lens[m] * 0.24, y=100, s=r'$\delta^\mathrm{Fiducial}_{\bar{\zeta}} = ' + r'{:.3f}$'.format(fiducial_delta_lens[m]), fontsize=15, color='darkred')
        
        plot[m, 0].fill_betweenx(y=[10, 800], x1=histogram_scatter_lens[m] - factor_lens[m], x2=histogram_scatter_lens[m] + factor_lens[m], color='gray', alpha=0.5)
        
        plot[m, 0].set_ylim(10, 800)
        plot[m, 0].set_xlim(histogram_scatter_lens[m] - range_lens[m], histogram_scatter_lens[m] + range_lens[m])
        
        plot[m, 0].set_yscale('log')
        plot[m, 0].set_ylabel(r'$\psi ( \zeta )$')
        
        if m == 0:
            plot[m, 0].set_title(r'$\mathrm{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 0].set_xlabel(r'$\zeta$')
    
    for m in range(bin_size):
        plot[m, 1].hist(som_deviation_source[:, m], bins=size, range=(histogram_scatter_source[m] - range_source[m], histogram_scatter_source[m] + range_source[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(model_deviation_source[:, m], bins=size, range=(histogram_scatter_source[m] - range_source[m], histogram_scatter_source[m] + range_source[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(product_deviation_source[:, m], bins=size, range=(histogram_scatter_source[m] - range_source[m], histogram_scatter_source[m] + range_source[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(fiducial_deviation_source[:, m], bins=size, range=(histogram_scatter_source[m] - range_source[m], histogram_scatter_source[m] + range_source[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(histogram_deviation_source[:, m], bins=size, range=(histogram_scatter_source[m] - range_source[m], histogram_scatter_source[m] + range_source[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].text(x=histogram_scatter_source[m] - range_source[m] * 0.80, y=250, s=r'$\delta^\mathrm{SOM}_{\bar{\zeta}} = ' + r'{:.3f}$'.format(som_delta_source[m]), fontsize=15, color='darkblue')
        
        plot[m, 1].text(x=histogram_scatter_source[m] - range_source[m] * 0.80, y=100, s=r'$\delta^\mathrm{Model}_{\bar{\zeta}} = ' + r'{:.3f}$'.format(model_delta_source[m]), fontsize=15, color='darkgreen')
        
        plot[m, 1].text(x=histogram_scatter_source[m] + range_source[m] * 0.24, y=250, s=r'$\delta^\mathrm{Product}_{\bar{\zeta}} = ' + r'{:.3f}$'.format(product_delta_source[m]), fontsize=15, color='darkorange')
        
        plot[m, 1].text(x=histogram_scatter_source[m] + range_source[m] * 0.24, y=100, s=r'$\delta^\mathrm{Fiducial}_{\bar{\zeta}} = ' + r'{:.3f}$'.format(fiducial_delta_source[m]), fontsize=15, color='darkred')
        
        plot[m, 1].fill_betweenx(y=[10, 800], x1=histogram_scatter_source[m] - factor_source[m], x2=histogram_scatter_source[m] + factor_source[m], color='gray', alpha=0.5)
        
        plot[m, 1].set_ylim(10, 800)
        plot[m, 1].set_xlim(histogram_scatter_source[m] - range_source[m], histogram_scatter_source[m] + range_source[m])
        
        plot[m, 1].set_yscale('log')
        plot[m, 1].set_ylabel('')
        
        if m == 0:
            plot[m, 1].set_title(r'$\mathrm{Source}$')
        
        if m == bin_size - 1:
            plot[m, 1].set_xlabel(r'$\zeta$')
    
    figure.subplots_adjust(wspace=0.12, hspace=0.24)
    figure.savefig(os.path.join(analyze_folder, '{}/DEVIATION/FIGURE_{}.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Analyze Expectation')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, LABEL, FOLDER)
