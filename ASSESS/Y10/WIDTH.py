import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot


def main(tag, label, number, folder):
    '''
    Plot the width of the lens and source redshift distributions
    
    Arguments:
        tag (str): The tag of the configuration
        label (str): The label of the configuration
        number (int): The number of configurations
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    print('Label: {}'.format(label))
    
    # Path
    assess_folder = os.path.join(folder, 'ASSESS/')
    os.makedirs(os.path.join(assess_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(assess_folder, '{}/WIDTH/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(assess_folder, '{}/WIDTH/{}/'.format(tag, label)), exist_ok=True)
    
    # Data
    gold_delta_lens = []
    gold_delta_source = []
    
    silver_delta_lens = []
    silver_delta_source = []
    
    copper_delta_lens = []
    copper_delta_source = []
    
    iron_delta_lens = []
    iron_delta_source = []
    
    titanium_delta_lens = []
    titanium_delta_source = []
    
    zinc_delta_lens = []
    zinc_delta_source = []
    
    # Value
    for index in range(number + 1):
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/GOLD/{}/DATA{}.hdf5'.format(tag, label, index)), 'r') as file:
            gold_eta_lens = file['lens']['average_eta'][...]
            gold_eta_source = file['source']['average_eta'][...]
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/GOLD/TRUTH/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            truth_mu_lens = file['lens']['average_mu'][...]
            truth_mu_source = file['source']['average_mu'][...]
            
            truth_eta_lens = file['lens']['average_eta'][...]
            truth_eta_source = file['source']['average_eta'][...]
        
        gold_delta_lens.append((gold_eta_lens - truth_eta_lens) / (1 + truth_mu_lens))
        gold_delta_source.append((gold_eta_source - truth_eta_source) / (1 + truth_mu_source))
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/SILVER/{}/DATA{}.hdf5'.format(tag, label, index)), 'r') as file:
            silver_eta_lens = file['lens']['average_eta'][...]
            silver_eta_source = file['source']['average_eta'][...]
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/SILVER/TRUTH/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            truth_mu_lens = file['lens']['average_mu'][...]
            truth_mu_source = file['source']['average_mu'][...]
            
            truth_eta_lens = file['lens']['average_eta'][...]
            truth_eta_source = file['source']['average_eta'][...]
        
        silver_delta_lens.append((silver_eta_lens - truth_eta_lens) / (1 + truth_mu_lens))
        silver_delta_source.append((silver_eta_source - truth_eta_source) / (1 + truth_mu_source))
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/COPPER/{}/DATA{}.hdf5'.format(tag, label, index)), 'r') as file:
            copper_eta_lens = file['lens']['average_eta'][...]
            copper_eta_source = file['source']['average_eta'][...]
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/COPPER/TRUTH/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            truth_mu_lens = file['lens']['average_mu'][...]
            truth_mu_source = file['source']['average_mu'][...]
            
            truth_eta_lens = file['lens']['average_eta'][...]
            truth_eta_source = file['source']['average_eta'][...]
        
        copper_delta_lens.append((copper_eta_lens - truth_eta_lens) / (1 + truth_mu_lens))
        copper_delta_source.append((copper_eta_source - truth_eta_source) / (1 + truth_mu_source))
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/IRON/{}/DATA{}.hdf5'.format(tag, label, index)), 'r') as file:
            iron_eta_lens = file['lens']['average_eta'][...]
            iron_eta_source = file['source']['average_eta'][...]
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/IRON/TRUTH/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            truth_mu_lens = file['lens']['average_mu'][...]
            truth_mu_source = file['source']['average_mu'][...]
            
            truth_eta_lens = file['lens']['average_eta'][...]
            truth_eta_source = file['source']['average_eta'][...]
        
        iron_delta_lens.append((iron_eta_lens - truth_eta_lens) / (1 + truth_mu_lens))
        iron_delta_source.append((iron_eta_source - truth_eta_source) / (1 + truth_mu_source))
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/TITANIUM/{}/DATA{}.hdf5'.format(tag, label, index)), 'r') as file:
            titanium_eta_lens = file['lens']['average_eta'][...]
            titanium_eta_source = file['source']['average_eta'][...]
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/TITANIUM/TRUTH/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            truth_mu_lens = file['lens']['average_mu'][...]
            truth_mu_source = file['source']['average_mu'][...]
            
            truth_eta_lens = file['lens']['average_eta'][...]
            truth_eta_source = file['source']['average_eta'][...]
        
        titanium_delta_lens.append((titanium_eta_lens - truth_eta_lens) / (1 + truth_mu_lens))
        titanium_delta_source.append((titanium_eta_source - truth_eta_source) / (1 + truth_mu_source))
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/ZINC/{}/DATA{}.hdf5'.format(tag, label, index)), 'r') as file:
            zinc_eta_lens = file['lens']['average_eta'][...]
            zinc_eta_source = file['source']['average_eta'][...]
        
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/ZINC/TRUTH/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            truth_mu_lens = file['lens']['average_mu'][...]
            truth_mu_source = file['source']['average_mu'][...]
            
            truth_eta_lens = file['lens']['average_eta'][...]
            truth_eta_source = file['source']['average_eta'][...]
        
        zinc_delta_lens.append((zinc_eta_lens - truth_eta_lens) / (1 + truth_mu_lens))
        zinc_delta_source.append((zinc_eta_source - truth_eta_source) / (1 + truth_mu_source))
    
    # Delta
    gold_delta_lens = numpy.array(gold_delta_lens)
    gold_delta_source = numpy.array(gold_delta_source)
    
    silver_delta_lens = numpy.array(silver_delta_lens)
    silver_delta_source = numpy.array(silver_delta_source)
    
    copper_delta_lens = numpy.array(copper_delta_lens)
    copper_delta_source = numpy.array(copper_delta_source)
    
    iron_delta_lens = numpy.array(iron_delta_lens)
    iron_delta_source = numpy.array(iron_delta_source)
    
    titanium_delta_lens = numpy.array(titanium_delta_lens)
    titanium_delta_source = numpy.array(titanium_delta_source)
    
    zinc_delta_lens = numpy.array(zinc_delta_lens)
    zinc_delta_source = numpy.array(zinc_delta_source)
    
    # Variable
    factor_lens = 0.005
    range_lens = [0.020, 0.020, 0.025, 0.025, 0.030, 0.030, 0.035, 0.035, 0.040, 0.040]
    
    factor_source = 0.002
    range_source = [0.045, 0.050, 0.055, 0.060, 0.065]
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 30
    
    # Figure
    bin_size = 5
    name_list = ['Gold', 'Silver', 'Copper', 'Iron', 'Titanium', 'Zinc']
    colors = {'DIR': 'darkmagenta', 'STACK': 'darkgreen', 'HYBRID': 'darkorange'}
    figure, plot = pyplot.subplots(nrows=bin_size, ncols=3, figsize=(20, 5 * bin_size))
    
    # Plot Lens
    for m in range(bin_size):
        violin = plot[m, 0].violinplot(
            vert=True, 
            widths=0.8,
            showmeans=True,
            showmedians=False,
            showextrema=True,
            positions=[1, 2, 3, 4, 5, 6],
            dataset=[gold_delta_lens[:, m], silver_delta_lens[:, m], copper_delta_lens[:, m], iron_delta_lens[:, m], titanium_delta_lens[:, m], zinc_delta_lens[:, m]]
        )
        
        for n in range(len(name_list)):
            violin['bodies'][n].set_alpha(0.60)
            violin['bodies'][n].set_facecolor(colors[label])
        
        violin['cbars'].set_color('black')
        violin['cmins'].set_color('black')
        violin['cmaxes'].set_color('black')
        violin['cmeans'].set_color('black')
        
        plot[m, 0].axhspan(-factor_lens, factor_lens, alpha=0.3, color='gray')
        plot[m, 0].text(x=5.5, y=range_lens[m] / 3 * 2, s=r'$\mathrm{Bin \,}' + r'{:.0f}$'.format(m + 1), color='black', ha='center')
        
        plot[m, 0].set_xlim(0.5, 6.5)
        plot[m, 0].set_ylim(-range_lens[m], +range_lens[m])
        
        plot[m, 0].set_ylabel(r'$\delta_\eta$')
        plot[m, 0].set_xticks([1, 2, 3, 4, 5, 6])
        plot[m, 0].tick_params(axis='y', labelsize=20)
        
        if m == 0:
            plot[m, 0].set_title(r'$\mathrm{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 0].set_xticklabels([r'$\mathrm{' + name + '}$' for name in name_list], rotation=45, ha='right', fontsize=25)
        else:
            plot[m, 0].set_xticklabels([])
    
    # Plot Lens
    for m in range(bin_size):
        violin = plot[m, 1].violinplot(
            vert=True, 
            widths=0.8,
            showmeans=True,
            showmedians=False,
            showextrema=True,
            positions=[1, 2, 3, 4, 5, 6],
            dataset=[gold_delta_lens[:, m + bin_size], silver_delta_lens[:, m + bin_size], copper_delta_lens[:, m + bin_size], iron_delta_lens[:, m + bin_size], titanium_delta_lens[:, m + bin_size], zinc_delta_lens[:, m + bin_size]]
        )
        
        for n in range(len(name_list)):
            violin['bodies'][n].set_alpha(0.60)
            violin['bodies'][n].set_facecolor(colors[label])
        
        violin['cbars'].set_color('black')
        violin['cmins'].set_color('black')
        violin['cmaxes'].set_color('black')
        violin['cmeans'].set_color('black')
        
        plot[m, 1].axhspan(-factor_lens, factor_lens, alpha=0.3, color='gray')
        plot[m, 1].text(x=5.5, y=range_lens[m + bin_size] / 3 * 2, s=r'$\mathrm{Bin \,}' + r'{:.0f}$'.format(m + bin_size + 1), color='black', ha='center')
        
        plot[m, 1].set_xlim(0.5, 6.5)
        plot[m, 1].set_ylim(-range_lens[m + bin_size], +range_lens[m + bin_size])
        
        plot[m, 1].set_ylabel(r'$\delta_\eta$')
        plot[m, 1].set_xticks([1, 2, 3, 4, 5, 6])
        plot[m, 1].tick_params(axis='y', labelsize=20)
        
        if m == 0:
            plot[m, 1].set_title(r'$\mathrm{Lens}$')
        
        if m == bin_size - 1:
            plot[m, 1].set_xticklabels([r'$\mathrm{' + name + '}$' for name in name_list], rotation=45, ha='right', fontsize=25)
        else:
            plot[m, 1].set_xticklabels([])
    
    # Plot Source
    for m in range(bin_size):
        violin = plot[m, 2].violinplot(
            vert=True, 
            widths=0.8,
            showmeans=True,
            showmedians=False,
            showextrema=True,
            positions=[1, 2, 3, 4, 5, 6],
            dataset=[gold_delta_source[:, m], silver_delta_source[:, m], copper_delta_source[:, m], iron_delta_source[:, m], titanium_delta_source[:, m], zinc_delta_source[:, m]]
        )
        
        for n in range(len(name_list)):
            violin['bodies'][n].set_alpha(0.60)
            violin['bodies'][n].set_facecolor(colors[label])
        
        violin['cbars'].set_color('black')
        violin['cmins'].set_color('black')
        violin['cmaxes'].set_color('black')
        violin['cmeans'].set_color('black')
        
        plot[m, 2].axhspan(-factor_source, factor_source, alpha=0.3, color='gray')
        plot[m, 2].text(x=5.5, y=range_source[m] / 3 * 2, s=r'$\mathrm{Bin \,}' + r'{:.0f}$'.format(m + 1), color='black', ha='center')
        
        plot[m, 2].set_xlim(0.5, 6.5)
        plot[m, 2].set_ylim(-range_source[m], +range_source[m])
        
        plot[m, 2].set_xticks([1, 2, 3, 4, 5, 6])
        plot[m, 2].tick_params(axis='y', labelsize=20)
        
        if m == 0:
            plot[m, 2].set_title(r'$\mathrm{Source}$')
        
        if m == bin_size - 1:
            plot[m, 2].set_xticklabels([r'$\mathrm{' + name + '}$' for name in name_list], rotation=45, ha='right', fontsize=25)
        else:
            plot[m, 2].set_xticklabels([])
    
    figure.subplots_adjust(wspace=0.24, hspace=0.08)
    figure.savefig(os.path.join(assess_folder, '{}/WIDTH/{}/FIGURE.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Assess Width')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--number', type=int, required=True, help='The number of configurations')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    LABEL = PARSE.parse_args().label
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, LABEL, NUMBER, FOLDER)