import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot


def main(tag, name, number, folder):
    '''
    Plot the deviation of the lens and source redshift distributions
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
        number (int): The number of configurations
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    print('Name: {}'.format(name))
    
    # Path
    assess_folder = os.path.join(folder, 'ASSESS/')
    os.makedirs(os.path.join(assess_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(assess_folder, '{}/DEVIATION/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(assess_folder, '{}/DEVIATION/{}/'.format(tag, name)), exist_ok=True)
    
    # Data
    dir_delta_lens = []
    dir_delta_source = []
    
    hybrid_delta_lens = []
    hybrid_delta_source = []
    
    stack_delta_lens = []
    stack_delta_source = []
    
    # Value
    for index in range(number + 1):
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/{}/DIR/DATA{}.hdf5'.format(tag, name, index)), 'r') as file:
            dir_eta_lens = file['lens']['average_eta'][...]
            dir_eta_source = file['source']['average_eta'][...]
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/{}/HYBRID/DATA{}.hdf5'.format(tag, name, index)), 'r') as file:
            hybrid_eta_lens = file['lens']['average_eta'][...]
            hybrid_eta_source = file['source']['average_eta'][...]
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/{}/STACK/DATA{}.hdf5'.format(tag, name, index)), 'r') as file:
            stack_eta_lens = file['lens']['average_eta'][...]
            stack_eta_source = file['source']['average_eta'][...]
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/{}/TRUTH/DATA{}.hdf5'.format(tag, name, index)), 'r') as file:
            truth_eta_lens = file['lens']['average_eta'][...]
            truth_eta_source = file['source']['average_eta'][...]
            
            truth_mu_lens = file['lens']['average_mu'][...]
            truth_mu_source = file['source']['average_mu'][...]
        
        dir_delta_lens.append((dir_eta_lens - truth_eta_lens) / (1 + truth_mu_lens))
        dir_delta_source.append((dir_eta_source - truth_eta_source) / (1 + truth_mu_source))
        
        hybrid_delta_lens.append((hybrid_eta_lens - truth_eta_lens) / (1 + truth_mu_lens))
        hybrid_delta_source.append((hybrid_eta_source - truth_eta_source) / (1 + truth_mu_source))
        
        stack_delta_lens.append((stack_eta_lens - truth_eta_lens) / (1 + truth_mu_lens))
        stack_delta_source.append((stack_eta_source - truth_eta_source) / (1 + truth_mu_source))
    
    # Delta
    dir_delta_lens = numpy.array(dir_delta_lens)
    dir_delta_source = numpy.array(dir_delta_source)
    
    hybrid_delta_lens = numpy.array(hybrid_delta_lens)
    hybrid_delta_source = numpy.array(hybrid_delta_source)
    
    stack_delta_lens = numpy.array(stack_delta_lens)
    stack_delta_source = numpy.array(stack_delta_source)
    
    # Variable
    factor_lens = 0.005
    range_lens = [0.030, 0.030, 0.035, 0.035, 0.040, 0.040, 0.045, 0.045, 0.050, 0.050]
    
    factor_source = 0.002
    range_source = [0.055, 0.060, 0.065, 0.070, 0.075]
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 30
    
    # Figure
    bin_size = 5
    label_list = ['DIR', 'Stack', 'Hybrid']
    colors = {'DIR': 'darkmagenta', 'Stack': 'darkgreen', 'Hybrid': 'darkorange'}
    figure, plot = pyplot.subplots(nrows=bin_size, ncols=3, figsize=(20, 5 * bin_size))
    
    # Plot Lens
    for m in range(bin_size):
        violin = plot[m, 0].violinplot(
            widths=0.8,
            vert=False, 
            showmeans=False, 
            showmedians=True,
            showextrema=True,
            positions=[3, 2, 1], 
            dataset=[dir_delta_lens[:, m], stack_delta_lens[:, m], hybrid_delta_lens[:, m]]
        )
        
        for n, color in enumerate(colors[label] for label in label_list):
            violin['bodies'][n].set_alpha(0.60)
            violin['bodies'][n].set_facecolor(color)
        
        violin['cbars'].set_color('black')
        violin['cmins'].set_color('black')
        violin['cmaxes'].set_color('black')
        violin['cmedians'].set_color('black')
        
        plot[m, 0].axvspan(-factor_lens, factor_lens, alpha=0.3, color='gray')
        plot[m, 0].text(x=range_lens[m] / 3 * 2, y=2.25, s=r'$\mathrm{Bin \,}' + r'{:.0f}$'.format(m + 1), color='black', ha='center')
        
        plot[m, 0].set_ylim(0.5, 3.5)
        plot[m, 0].tick_params(axis='x', labelsize=25)
        plot[m, 0].set_xlim(-range_lens[m], +range_lens[m])
        
        plot[m, 0].set_yticks([3, 2, 1])
        plot[m, 0].set_yticklabels([r'$\mathrm{' + label + '}$' for label in label_list])
        
        if m == 0:
            plot[m, 0].set_title(r'$\mathrm{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 0].set_xlabel(r'$\delta_\eta$')
    
    # Plot Lens
    for m in range(bin_size):
        violin = plot[m, 1].violinplot(
            widths=0.8,
            vert=False, 
            showmeans=False, 
            showmedians=True,
            showextrema=True,
            positions=[3, 2, 1], 
            dataset=[dir_delta_lens[:, m + bin_size], stack_delta_lens[:, m + bin_size], hybrid_delta_lens[:, m + bin_size]]
        )
        
        for n, color in enumerate(colors[label] for label in label_list):
            violin['bodies'][n].set_alpha(0.60)
            violin['bodies'][n].set_facecolor(color)
        
        violin['cbars'].set_color('black')
        violin['cmins'].set_color('black')
        violin['cmaxes'].set_color('black')
        violin['cmedians'].set_color('black')
        
        plot[m, 1].axvspan(-factor_lens, factor_lens, alpha=0.3, color='gray')
        plot[m, 1].text(x=range_lens[m + bin_size] / 3 * 2, y=2.25, s=r'$\mathrm{Bin \,}' + r'{:.0f}$'.format(m + bin_size + 1), color='black', ha='center')
        
        plot[m, 1].set_ylim(0.5, 3.5)
        plot[m, 1].tick_params(axis='x', labelsize=25)
        plot[m, 1].set_xlim(-range_lens[m + bin_size], +range_lens[m + bin_size])
        
        plot[m, 1].set_yticks([3, 2, 1])
        plot[m, 1].set_yticklabels([])
        
        if m == 0:
            plot[m, 1].set_title(r'$\mathrm{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 1].set_xlabel(r'$\delta_\eta$')
    
    # Plot Source
    for m in range(bin_size):
        violin = plot[m, 2].violinplot(
            widths=0.8,
            vert=False, 
            showmeans=False,
            showmedians=True,
            showextrema=True,
            positions=[3, 2, 1],
            dataset=[dir_delta_source[:, m], stack_delta_source[:, m], hybrid_delta_source[:, m]]
        )
        
        for n, color in enumerate(colors[label] for label in label_list):
            violin['bodies'][n].set_alpha(0.60)
            violin['bodies'][n].set_facecolor(color)
        
        violin['cbars'].set_color('black')
        violin['cmins'].set_color('black')
        violin['cmaxes'].set_color('black')
        violin['cmedians'].set_color('black')
        
        plot[m, 2].axvspan(-factor_source, factor_source, alpha=0.3, color='gray')
        plot[m, 2].text(x=range_source[m] / 3 * 2, y=2.25, s=r'$\mathrm{Bin \,}' + r'{:.0f}$'.format(m + 1), color='black', ha='center')
        
        plot[m, 2].set_ylim(0.5, 3.5)
        plot[m, 2].tick_params(axis='x', labelsize=25)
        plot[m, 2].set_xlim(-range_source[m], +range_source[m])
        
        plot[m, 2].set_yticklabels([])
        plot[m, 2].set_yticks([3, 2, 1])
        
        if m == 0:
            plot[m, 2].set_title(r'$\mathrm{Source}$')
        
        if m == bin_size - 1:
            plot[m, 2].set_xlabel(r'$\delta_\eta$')
    
    figure.subplots_adjust(wspace=0.12, hspace=0.12)
    figure.savefig(os.path.join(assess_folder, '{}/DEVIATION/{}/FIGURE.pdf'.format(tag, name)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Assess Deviation')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the configuration')
    PARSE.add_argument('--number', type=int, required=True, help='The number of configurations')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, NUMBER, FOLDER)