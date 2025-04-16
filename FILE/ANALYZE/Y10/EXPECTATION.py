import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot


def main(tag, label, folder):
    '''
    Plot the expectation of the lens and source redshift distributions
    
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
    os.makedirs(os.path.join(analyze_folder, '{}/EXPECTATION/'.format(tag)), exist_ok=True)
    
    # Info
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/HISTOGRAM_{}.hdf5'.format(tag, label)), 'r') as file:
        histogram_middle_lens = file['lens']['middle'][...]
        histogram_middle_source = file['source']['middle'][...]
        
        histogram_expectation_lens = file['lens']['expectation'][...]
        histogram_expectation_source = file['source']['expectation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/MODEL_{}.hdf5'.format(tag, label)), 'r') as file:
        model_middle_lens = file['lens']['middle'][...]
        model_middle_source = file['source']['middle'][...]
        
        model_expectation_lens = file['lens']['expectation'][...]
        model_expectation_source = file['source']['expectation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/PRODUCT_{}.hdf5'.format(tag, label)), 'r') as file:
        product_middle_lens = file['lens']['middle'][...]
        product_middle_source = file['source']['middle'][...]
        
        product_expectation_lens = file['lens']['expectation'][...]
        product_expectation_source = file['source']['expectation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/FIDUCIAL_{}.hdf5'.format(tag, label)), 'r') as file:
        fiducial_middle_lens = file['lens']['middle'][...]
        fiducial_middle_source = file['source']['middle'][...]
        
        fiducial_expectation_lens = file['lens']['expectation'][...]
        fiducial_expectation_source = file['source']['expectation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/TARGET_{}.hdf5'.format(tag, label)), 'r') as file:
        target_middle_lens = file['lens']['middle'][...]
        target_middle_source = file['source']['middle'][...]
        
        target_expectation_lens = file['lens']['expectation'][...]
        target_expectation_source = file['source']['expectation'][...]
    
    # Delta
    histogram_delta_lens = numpy.abs(histogram_middle_lens - target_middle_lens) / (1 + target_middle_lens)
    histogram_delta_source = numpy.abs(histogram_middle_source - target_middle_source) / (1 + target_middle_source)
    
    model_delta_lens = numpy.abs(model_middle_lens - target_middle_lens) / (1 + target_middle_lens)
    model_delta_source = numpy.abs(model_middle_source - target_middle_source) / (1 + target_middle_source)
    
    product_delta_lens = numpy.abs(product_middle_lens - target_middle_lens) / (1 + target_middle_lens)
    product_delta_source = numpy.abs(product_middle_source - target_middle_source) / (1 + target_middle_source)
    
    fiducial_delta_lens = numpy.abs(fiducial_middle_lens - target_middle_lens) / (1 + target_middle_lens)
    fiducial_delta_source = numpy.abs(fiducial_middle_source - target_middle_source) / (1 + target_middle_source)
    
    # Variable
    size = 100
    bin_size = 5
    
    range_lens = 0.025 * (1 + target_middle_lens)
    range_source = 0.050 * (1 + target_middle_source)
    
    factor_lens = 0.003 * (1 + target_middle_lens)
    factor_source = 0.001 * (1 + target_middle_source)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot
    figure, plot = pyplot.subplots(nrows=bin_size, ncols=3, figsize=(18, 3 * bin_size))
    
    for m in range(bin_size):
        
        plot[m, 0].hist(histogram_expectation_lens[:, m], bins=size, range=(target_middle_lens[m] - range_lens[m], target_middle_lens[m] + range_lens[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(model_expectation_lens[:, m], bins=size, range=(target_middle_lens[m] - range_lens[m], target_middle_lens[m] + range_lens[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(product_expectation_lens[:, m], bins=size, range=(target_middle_lens[m] - range_lens[m], target_middle_lens[m] + range_lens[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(fiducial_expectation_lens[:, m], bins=size, range=(target_middle_lens[m] - range_lens[m], target_middle_lens[m] + range_lens[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(target_expectation_lens[:, m], bins=size, range=(target_middle_lens[m] - range_lens[m], target_middle_lens[m] + range_lens[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].text(x=target_middle_lens[m] - range_lens[m] * 0.85, y=500, s=r'$\delta^\mathtt{histogram}_{\tilde{\mu}} = ' + r'{:.3f}$'.format(histogram_delta_lens[m]), fontsize=15, color='darkblue')
        
        plot[m, 0].text(x=target_middle_lens[m] - range_lens[m] * 0.85, y=200, s=r'$\delta^\mathtt{model}_{\tilde{\mu}} = ' + r'{:.3f}$'.format(model_delta_lens[m]), fontsize=15, color='darkgreen')
        
        plot[m, 0].text(x=target_middle_lens[m] + range_lens[m] * 0.25, y=500, s=r'$\delta^\mathtt{product}_{\tilde{\mu}} = ' + r'{:.3f}$'.format(product_delta_lens[m]), fontsize=15, color='darkorange')
        
        plot[m, 0].text(x=target_middle_lens[m] + range_lens[m] * 0.25, y=200, s=r'$\delta^\mathtt{fiducial}_{\tilde{\mu}} = ' + r'{:.3f}$'.format(fiducial_delta_lens[m]), fontsize=15, color='darkred')
        
        plot[m, 0].fill_betweenx(y=[10, 800], x1=target_middle_lens[m] - factor_lens[m], x2=target_middle_lens[m] + factor_lens[m], color='gray', alpha=0.5)
        
        plot[m, 0].set_ylim(10, 800)
        plot[m, 0].set_xlim(target_middle_lens[m] - range_lens[m], target_middle_lens[m] + range_lens[m])
        
        plot[m, 0].set_yscale('log')
        plot[m, 0].set_ylabel(r'$\psi ( \mu )$')
        
        if m == 0:
            plot[m, 0].set_title(r'$\mathrm{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 0].set_xlabel(r'$\mu$')
    
    for m in range(bin_size):
        
        plot[m, 1].hist(histogram_expectation_lens[:, m + bin_size], bins=size, range=(target_middle_lens[m + bin_size] - range_lens[m], target_middle_lens[m + bin_size] + range_lens[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(model_expectation_lens[:, m + bin_size], bins=size, range=(target_middle_lens[m + bin_size] - range_lens[m], target_middle_lens[m + bin_size] + range_lens[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(product_expectation_lens[:, m + bin_size], bins=size, range=(target_middle_lens[m + bin_size] - range_lens[m], target_middle_lens[m + bin_size] + range_lens[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(fiducial_expectation_lens[:, m + bin_size], bins=size, range=(target_middle_lens[m + bin_size] - range_lens[m], target_middle_lens[m + bin_size] + range_lens[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(target_expectation_lens[:, m + bin_size], bins=size, range=(target_middle_lens[m + bin_size] - range_lens[m], target_middle_lens[m + bin_size] + range_lens[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].text(x=target_middle_lens[m + bin_size] - range_lens[m] * 0.85, y=500, s=r'$\delta^\mathtt{histogram}_{\tilde{\mu}} = ' + r'{:.3f}$'.format(histogram_delta_lens[m + bin_size]), fontsize=15, color='darkblue')
        
        plot[m, 1].text(x=target_middle_lens[m + bin_size] - range_lens[m] * 0.85, y=200, s=r'$\delta^\mathtt{model}_{\tilde{\mu}} = ' + r'{:.3f}$'.format(model_delta_lens[m + bin_size]), fontsize=15, color='darkgreen')
        
        plot[m, 1].text(x=target_middle_lens[m + bin_size] + range_lens[m] * 0.25, y=500, s=r'$\delta^\mathtt{product}_{\tilde{\mu}} = ' + r'{:.3f}$'.format(product_delta_lens[m + bin_size]), fontsize=15, color='darkorange')
        
        plot[m, 1].text(x=target_middle_lens[m + bin_size] + range_lens[m] * 0.25, y=200, s=r'$\delta^\mathtt{fiducial}_{\tilde{\mu}} = ' + r'{:.3f}$'.format(fiducial_delta_lens[m + bin_size]), fontsize=15, color='darkred')
        
        plot[m, 1].fill_betweenx(y=[10, 800], x1=target_middle_lens[m + bin_size] - factor_lens[m], x2=target_middle_lens[m + bin_size] + factor_lens[m], color='gray', alpha=0.5)
        
        plot[m, 1].set_ylim(10, 800)
        plot[m, 1].set_xlim(target_middle_lens[m + bin_size] - range_lens[m], target_middle_lens[m + bin_size] + range_lens[m])
        
        plot[m, 1].set_yscale('log')
        plot[m, 1].set_ylabel('')
        
        if m == 0:
            plot[m, 1].set_title(r'$\mathrm{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 1].set_xlabel(r'$\mu$')
    
    for m in range(bin_size):
        plot[m, 2].hist(histogram_expectation_source[:, m], bins=size, range=(target_middle_source[m] - range_source[m], target_middle_source[m] + range_source[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 2].hist(model_expectation_source[:, m], bins=size, range=(target_middle_source[m] - range_source[m], target_middle_source[m] + range_source[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 2].hist(product_expectation_source[:, m], bins=size, range=(target_middle_source[m] - range_source[m], target_middle_source[m] + range_source[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 2].hist(fiducial_expectation_source[:, m], bins=size, range=(target_middle_source[m] - range_source[m], target_middle_source[m] + range_source[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 2].hist(target_expectation_source[:, m], bins=size, range=(target_middle_source[m] - range_source[m], target_middle_source[m] + range_source[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 2].text(x=target_middle_source[m] - range_source[m] * 0.85, y=500, s=r'$\delta^\mathtt{histogram}_{\tilde{\mu}} = ' + r'{:.3f}$'.format(histogram_delta_source[m]), fontsize=15, color='darkblue')
        
        plot[m, 2].text(x=target_middle_source[m] - range_source[m] * 0.85, y=200, s=r'$\delta^\mathtt{model}_{\tilde{\mu}} = ' + r'{:.3f}$'.format(model_delta_source[m]), fontsize=15, color='darkgreen')
        
        plot[m, 2].text(x=target_middle_source[m] + range_source[m] * 0.25, y=500, s=r'$\delta^\mathtt{product}_{\tilde{\mu}} = ' + r'{:.3f}$'.format(product_delta_source[m]), fontsize=15, color='darkorange')
        
        plot[m, 2].text(x=target_middle_source[m] + range_source[m] * 0.25, y=200, s=r'$\delta^\mathtt{fiducial}_{\tilde{\mu}} = ' + r'{:.3f}$'.format(fiducial_delta_source[m]), fontsize=15, color='darkred')
        
        plot[m, 2].fill_betweenx(y=[10, 800], x1=target_middle_source[m] - factor_source[m], x2=target_middle_source[m] + factor_source[m], color='gray', alpha=0.5)
        
        plot[m, 2].set_ylim(10, 800)
        plot[m, 2].set_xlim(target_middle_source[m] - range_source[m], target_middle_source[m] + range_source[m])
        
        plot[m, 2].set_yscale('log')
        plot[m, 2].set_ylabel('')
        
        if m == 0:
            plot[m, 2].set_title(r'$\mathrm{Source}$')
        
        if m == bin_size - 1:
            plot[m, 2].set_xlabel(r'$\mu$')
    
    os.makedirs(analyze_folder, exist_ok=True)
    os.makedirs(os.path.join(analyze_folder, '{}/EXPECTATION/'.format(tag)), exist_ok=True)
    
    figure.subplots_adjust(wspace=0.12, hspace=0.24)
    figure.savefig(os.path.join(analyze_folder, '{}/EXPECTATION/FIGURE_{}.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
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