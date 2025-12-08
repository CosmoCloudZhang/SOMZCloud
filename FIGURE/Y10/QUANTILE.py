import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot


def main(tag, index, folder):
    '''
    Plot the figures of the quantiles
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        float: The duration of the plotter
    '''
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    figure_folder = os.path.join(folder, 'FIGURE/')
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/QUANTILE/'.format(tag)), exist_ok=True)
    
    # Target
    with h5py.File(os.path.join(model_folder, '{}/TARGET/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        z_quantile = file['z_quantile'][...]
        
        target_lens = file['target_lens'][...]
        target_source = file['target_source'][...]
        
        histogram_lens = file['histogram_lens'][...]
        histogram_source = file['histogram_source'][...]
    
    # Quantile
    z_quantile_lens = z_quantile[target_lens]
    z_quantile_source = z_quantile[target_source]
    
    # Histogram
    average_size_lens, histogram_size_lens = histogram_lens.shape
    average_size_source, histogram_size_source = histogram_source.shape
    
    histogram_bin_lens = numpy.linspace(0.0, 1.0, histogram_size_lens + 1)
    histogram_bin_source = numpy.linspace(0.0, 1.0, histogram_size_source + 1)
    
    # Average
    z1_average_lens = 0.0
    z2_average_lens = 1.5
    z_average_lens = numpy.linspace(z1_average_lens, z2_average_lens, average_size_lens + 1)
    
    z1_average_source = 0.0
    z2_average_source = 3.0
    z_average_source = numpy.linspace(z1_average_source, z2_average_source, average_size_source + 1)
    
    # Plot
    os.environ['PATH'] = '/pscratch/sd/y/yhzhang/texlive/2025/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 30
    
    # Figure
    figure, plot = pyplot.subplots(nrows=1, ncols=2, figsize=(16, 10))
    lens_color_list = ['darkmagenta', 'darkblue', 'darkgreen', 'darkorange', 'darkred']
    source_color_list = ['hotpink', 'darkorchid', 'deepskyblue', 'darkcyan', 'darkgoldenrod', 'darksalmon']
    
    # Lens
    for m in range(average_size_lens):
        plot[0].errorbar(
            alpha=0.6,
            marker='s',
            markersize=10,
            linewidth=3.0, 
            linestyle='-', 
            y=histogram_lens[m, :], 
            color=lens_color_list[m], 
            x=(histogram_bin_lens[+1:] + histogram_bin_lens[:-1]) / 2, 
            label=r'$z_\mathrm{true} \in ' + r'\left[ {:.1f}, {:.1f} \right]$'.format(z_average_lens[m], z_average_lens[m + 1])
        )
    
    plot[0].hist(z_quantile_lens, bins=histogram_bin_lens, density=True, color='black', histtype='step', linewidth=5.0, linestyle='-', label=r'$\mathrm{All}$')
    
    plot[0].axhline(y=1.0, xmin=0.0, xmax=1.0, color='black', linestyle=':', linewidth=5.0)
    
    plot[0].set_yscale('log')
    plot[0].set_xscale('linear')
    
    plot[0].set_xlim(0.0, 1.0)
    plot[0].set_ylim(0.08, 8.00)
    plot[0].set_xticks([0.0,0.2, 0.4, 0.6, 0.8, 1.0])
    
    plot[0].set_title(r'$\mathrm{Lens}$')
    plot[0].set_xlabel(r'$q \left( z_\mathrm{true} \right)$')
    plot[0].set_ylabel(r'$\phi \left[ q \left( z_\mathrm{true} \right) \right]$')
    
    # Legend
    handles, labels = plot[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='center', ncol=2, fontsize=20, bbox_to_anchor=(0.32, 0.0), frameon=True)
    
    # Source
    for m in range(average_size_source):
        plot[1].errorbar(
            alpha=0.6,
            marker='s',
            markersize=10,
            linewidth=3.0, 
            linestyle='-', 
            y=histogram_source[m, :], 
            color=source_color_list[m], 
            x=(histogram_bin_source[+1:] + histogram_bin_source[:-1]) / 2, 
            label=r'$z_\mathrm{true} \in ' + r'\left[ {:.1f}, {:.1f} \right]$'.format(z_average_source[m], z_average_source[m + 1])
        )
    
    plot[1].hist(z_quantile_source, bins=histogram_bin_source, density=True, color='black', histtype='step', linewidth=5.0, linestyle='-', label=r'$\mathrm{All}$')
    
    plot[1].axhline(y=1.0, xmin=0.0, xmax=1.0, color='black', linestyle=':', linewidth=5.0)
    
    plot[1].set_yscale('log')
    plot[1].set_xscale('linear')
    
    plot[1].set_xlim(0.0, 1.0)
    plot[1].set_ylim(0.08, 8.00)
    
    plot[1].set_yticklabels([])
    plot[1].set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    
    plot[1].set_title(r'$\mathrm{Source}$')
    plot[1].set_xlabel(r'$q \left( z_\mathrm{true} \right)$')
    
    # Legend
    handles, labels = plot[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc='center', ncol=2, fontsize=20, bbox_to_anchor=(0.72, 0.0), frameon=True)
    
    # Save
    figure.subplots_adjust(wspace=0.0, hspace=0.0, bottom=0.225)
    figure.savefig(os.path.join(figure_folder, '{}/QUANTILE/FIGURE{}.pdf'.format(tag, index)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Quantile')
    PARSE.add_argument('--tag', type=str, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)