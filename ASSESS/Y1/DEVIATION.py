import os
import h5py
import time
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
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/DIR_{}.hdf5'.format(tag, label)), 'r') as file:
        dir_deviation_lens = file['lens']['deviation'][...]
        dir_deviation_source = file['source']['deviation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/STACK_{}.hdf5'.format(tag, label)), 'r') as file:
        stack_deviation_lens = file['lens']['deviation'][...]
        stack_deviation_source = file['source']['deviation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/PRODUCT_{}.hdf5'.format(tag, label)), 'r') as file:
        product_deviation_lens = file['lens']['deviation'][...]
        product_deviation_source = file['source']['deviation'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/TRUTH_{}.hdf5'.format(tag, label)), 'r') as file:
        truth_width_lens = file['lens']['width'][...]
        truth_width_source = file['source']['width'][...]
        
        truth_deviation_lens = file['lens']['deviation'][...]
        truth_deviation_source = file['source']['deviation'][...]
        
        truth_middle_lens = file['lens']['middle'][...]
        truth_middle_source = file['source']['middle'][...]
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 30
    
    # Variable
    size = 100
    bin_size = 5
    
    range_lens = 0.5 * truth_width_lens
    range_source = 1.0 * truth_width_source
    
    factor_lens = 0.003 * (1 + truth_middle_lens)
    factor_source = 0.001 * (1 + truth_middle_source)
    
    # Plot
    figure, plot = pyplot.subplots(nrows=bin_size, ncols=3, figsize=(18, 5 * bin_size))
    
    for m in range(bin_size):
        
        plot[m, 0].hist(dir_deviation_lens[:, m], bins=size, range=(truth_width_lens[m] - range_lens[m], truth_width_lens[m] + range_lens[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(stack_deviation_lens[:, m], bins=size, range=(truth_width_lens[m] - range_lens[m], truth_width_lens[m] + range_lens[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(product_deviation_lens[:, m], bins=size, range=(truth_width_lens[m] - range_lens[m], truth_width_lens[m] + range_lens[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(truth_deviation_lens[:, m], bins=size, range=(truth_width_lens[m] - range_lens[m], truth_width_lens[m] + range_lens[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].fill_betweenx(y=[10, 1000], x1=truth_width_lens[m] - factor_lens[m], x2=truth_width_lens[m] + factor_lens[m], color='gray', alpha=0.5)
        
        plot[m, 0].text(x=truth_width_lens[m] + range_lens[m] / 3, y=500, s=r'$\mathrm{Bin \,}' + r'{:.0f}$'.format(m + 1), color='black')
        
        plot[m, 0].set_ylim(10, 1000)
        plot[m, 0].set_xlim(truth_width_lens[m] - range_lens[m], truth_width_lens[m] + range_lens[m])
        
        plot[m, 0].set_yscale('log')
        plot[m, 0].set_ylabel(r'$\psi ( \eta )$')
        
        if m == 0:
            plot[m, 0].set_title(r'$\mathtt{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 0].set_xlabel(r'$\eta$')
    
    for m in range(bin_size):
        
        plot[m, 1].hist(dir_deviation_lens[:, m + bin_size], bins=size, range=(truth_width_lens[m + bin_size] - range_lens[m + bin_size], truth_width_lens[m + bin_size] + range_lens[m + bin_size]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(dir_deviation_lens[:, m + bin_size], bins=size, range=(truth_width_lens[m + bin_size] - range_lens[m + bin_size], truth_width_lens[m + bin_size] + range_lens[m + bin_size]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(stack_deviation_lens[:, m + bin_size], bins=size, range=(truth_width_lens[m + bin_size] - range_lens[m + bin_size], truth_width_lens[m + bin_size] + range_lens[m + bin_size]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(product_deviation_lens[:, m + bin_size], bins=size, range=(truth_width_lens[m + bin_size] - range_lens[m + bin_size], truth_width_lens[m + bin_size] + range_lens[m + bin_size]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].fill_betweenx(y=[10, 1000], x1=truth_width_lens[m + bin_size] - factor_lens[m + bin_size], x2=truth_width_lens[m + bin_size] + factor_lens[m + bin_size], color='gray', alpha=0.5)
        
        plot[m, 1].text(x=truth_width_lens[m + bin_size] + range_lens[m + bin_size] / 3, y=500, s=r'$\mathrm{Bin \,}' + r'{:.0f}$'.format(m + bin_size + 1), color='black')
        
        plot[m, 1].set_ylim(10, 1000)
        plot[m, 1].set_xlim(truth_width_lens[m + bin_size] - range_lens[m + bin_size], truth_width_lens[m + bin_size] + range_lens[m + bin_size])
        
        plot[m, 1].set_yscale('log')
        plot[m, 1].set_ylabel('')
        
        if m == 0:
            plot[m, 1].set_title(r'$\mathtt{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 1].set_xlabel(r'$\eta$')
    
    for m in range(bin_size):
        plot[m, 2].hist(dir_deviation_source[:, m], bins=size, range=(truth_width_source[m] - range_source[m], truth_width_source[m] + range_source[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 2].hist(stack_deviation_source[:, m], bins=size, range=(truth_width_source[m] - range_source[m], truth_width_source[m] + range_source[m]), color='darkgreen', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 2].hist(product_deviation_source[:, m], bins=size, range=(truth_width_source[m] - range_source[m], truth_width_source[m] + range_source[m]), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 2].hist(truth_deviation_source[:, m], bins=size, range=(truth_width_source[m] - range_source[m], truth_width_source[m] + range_source[m]), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 2].fill_betweenx(y=[4, 400], x1=truth_width_source[m] - factor_source[m], x2=truth_width_source[m] + factor_source[m], color='gray', alpha=0.5)
        
        plot[m, 2].text(x=truth_width_source[m] + range_source[m] / 3, y=200, s=r'$\mathrm{Bin \,}' + r'{:.0f}$'.format(m + 1), color='black')
        
        plot[m, 2].set_ylim(4, 400)
        plot[m, 2].set_xlim(truth_width_source[m] - range_source[m], truth_width_source[m] + range_source[m])
        
        plot[m, 2].set_yscale('log')
        plot[m, 2].set_ylabel('')
        
        if m == 0:
            plot[m, 2].set_title(r'$\mathtt{Source}$')
        
        if m == bin_size - 1:
            plot[m, 2].set_xlabel(r'$\eta$')
    
    figure.subplots_adjust(wspace=0.2, hspace=0.2)
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
    PARSE = argparse.ArgumentParser(description='Analyze Deviation')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, LABEL, FOLDER)