import os
import h5py
import time
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
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/DIR_{}.hdf5'.format(tag, label)), 'r') as file:
        dir_expectation_lens = file['lens']['expectation'][...]
        dir_expectation_source = file['source']['expectation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/STACK_{}.hdf5'.format(tag, label)), 'r') as file:
        stack_expectation_lens = file['lens']['expectation'][...]
        stack_expectation_source = file['source']['expectation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/PRODUCT_{}.hdf5'.format(tag, label)), 'r') as file:
        product_expectation_lens = file['lens']['expectation'][...]
        product_expectation_source = file['source']['expectation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/TRUTH_{}.hdf5'.format(tag, label)), 'r') as file:
        truth_middle_lens = file['lens']['middle'][...]
        truth_middle_source = file['source']['middle'][...]
        
        truth_expectation_lens = file['lens']['expectation'][...]
        truth_expectation_source = file['source']['expectation'][...]
    
    # Variable
    size = 100
    bin_size = 5
    
    range_lens = 0.015 * (1 + truth_middle_lens)
    range_source = 0.050 * (1 + truth_middle_source)
    
    factor_lens = 0.005 * (1 + truth_middle_lens)
    factor_source = 0.002 * (1 + truth_middle_source)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 30
    
    # Plot
    figure, plot = pyplot.subplots(nrows=bin_size, ncols=2, figsize=(12, 5 * bin_size))
    
    for m in range(bin_size):
        
        plot[m, 0].hist(dir_expectation_lens[:, m], bins=size, range=(truth_middle_lens[m] - range_lens[m], truth_middle_lens[m] + range_lens[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(stack_expectation_lens[:, m], bins=size, range=(truth_middle_lens[m] - range_lens[m], truth_middle_lens[m] + range_lens[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(product_expectation_lens[:, m], bins=size, range=(truth_middle_lens[m] - range_lens[m], truth_middle_lens[m] + range_lens[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(truth_expectation_lens[:, m], bins=size, range=(truth_middle_lens[m] - range_lens[m], truth_middle_lens[m] + range_lens[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].fill_betweenx(y=[5, 500], x1=truth_middle_lens[m] - factor_lens[m], x2=truth_middle_lens[m] + factor_lens[m], color='gray', alpha=0.5)
        
        plot[m, 0].text(x=truth_middle_lens[m] + range_lens[m] / 3, y=250, s=r'$\mathrm{Bin \,}' + r'{:.0f}$'.format(m + 1), color='black')
        
        plot[m, 0].set_ylim(5, 500)
        plot[m, 0].set_xlim(truth_middle_lens[m] - range_lens[m], truth_middle_lens[m] + range_lens[m])
        
        plot[m, 0].set_yscale('log')
        plot[m, 0].set_ylabel(r'$\psi \left( \mu \right)$')
        
        if m == 0:
            plot[m, 0].set_title(r'$\mathtt{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 0].set_xlabel(r'$\mu$')
    
    for m in range(bin_size):
        plot[m, 1].hist(dir_expectation_source[:, m], bins=size, range=(truth_middle_source[m] - range_source[m], truth_middle_source[m] + range_source[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(stack_expectation_source[:, m], bins=size, range=(truth_middle_source[m] - range_source[m], truth_middle_source[m] + range_source[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(product_expectation_source[:, m], bins=size, range=(truth_middle_source[m] - range_source[m], truth_middle_source[m] + range_source[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(truth_expectation_source[:, m], bins=size, range=(truth_middle_source[m] - range_source[m], truth_middle_source[m] + range_source[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].fill_betweenx(y=[2, 200], x1=truth_middle_source[m] - factor_source[m], x2=truth_middle_source[m] + factor_source[m], color='gray', alpha=0.5)
        
        plot[m, 1].text(x=truth_middle_source[m] + range_source[m] / 3, y=100, s=r'$\mathrm{Bin \,}' + r'{:.0f}$'.format(m + 1), color='black')
        
        plot[m, 1].set_ylim(2, 200)
        plot[m, 1].set_xlim(truth_middle_source[m] - range_source[m], truth_middle_source[m] + range_source[m])
        
        plot[m, 1].set_yscale('log')
        plot[m, 1].set_ylabel('')
        
        if m == 0:
            plot[m, 1].set_title(r'$\mathtt{Source}$')
        
        if m == bin_size - 1:
            plot[m, 1].set_xlabel(r'$\mu$')
    
    figure.subplots_adjust(wspace=0.2, hspace=0.2)
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