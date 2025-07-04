import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot
from matplotlib.gridspec import GridSpec


def main(tag, folder):
    '''
    Plot the figure of the color-redshift diagram
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    random_generator = numpy.random.default_rng(seed=0)
    
    # Path
    figure_folder = os.path.join(folder, 'FIGURE/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Load
    with h5py.File(os.path.join(dataset_folder, '{}/OBSERVATION/OBSERVATION.hdf5'.format(tag)), 'r') as file:
        observation_dataset = {key: file[key][...] for key in file.keys()}
    
    with h5py.File(os.path.join(dataset_folder, '{}/SIMULATION/SIMULATION.hdf5'.format(tag)), 'r') as file:
        simulation_dataset = {key: file[key][...] for key in file.keys()}
    
    # Filter
    magnitude1 = 16
    magnitude2 = 25
    
    filter = (magnitude1 < observation_dataset['mag_i_lsst']) & (observation_dataset['mag_i_lsst'] < magnitude2)
    observation_dataset = {key: observation_dataset[key][filter] for key in observation_dataset.keys()}
    
    # Simulation
    filter = (magnitude1 < simulation_dataset['mag_i_lsst']) & (simulation_dataset['mag_i_lsst'] < magnitude2)
    simulation_dataset = {key: simulation_dataset[key][filter] for key in simulation_dataset.keys()}
    
    # Association
    indices = random_generator.choice(len(simulation_dataset['redshift']), size=len(observation_dataset['redshift']), replace=True)
    simulation_dataset = {key: simulation_dataset[key][indices] for key in simulation_dataset.keys()}
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Redshift
    redshift1 = 0.05
    redshift2 = 2.95
    
    observation_redshift = observation_dataset['redshift_true']
    simulation_redshift = simulation_dataset['redshift_true']
    
    # Magnitude
    magnitude_edge1 = 22.0
    magnitude_edge2 = 23.0
    magnitude_edge3 = 24.0
    magnitude_edge = [magnitude1, magnitude_edge1, magnitude_edge2, magnitude_edge3, magnitude2]
    
    # Color
    color1 = [-0.4, -0.4, -0.4, -0.4]
    color2 = [+2.5, +2.0, +1.5, +1.0]
    label_list = [r'$u - g$', r'$g - r$', r'$i - z$', r'$z - y$']
    
    observation_color1 = observation_dataset['mag_u_lsst'] - observation_dataset['mag_g_lsst']
    observation_color2 = observation_dataset['mag_g_lsst'] - observation_dataset['mag_r_lsst']
    observation_color3 = observation_dataset['mag_i_lsst'] - observation_dataset['mag_z_lsst']
    observation_color4 = observation_dataset['mag_z_lsst'] - observation_dataset['mag_y_lsst']
    observation_color = [observation_color1, observation_color2, observation_color3, observation_color4]
    
    simulation_color1 = simulation_dataset['mag_u_lsst'] - simulation_dataset['mag_g_lsst']
    simulation_color2 = simulation_dataset['mag_g_lsst'] - simulation_dataset['mag_r_lsst']
    simulation_color3 = simulation_dataset['mag_i_lsst'] - simulation_dataset['mag_z_lsst']
    simulation_color4 = simulation_dataset['mag_z_lsst'] - simulation_dataset['mag_y_lsst']
    simulation_color = [simulation_color1, simulation_color2, simulation_color3, simulation_color4]
    
    # Figure
    figure = pyplot.figure(figsize=(20, 20))
    gridspec = GridSpec(nrows=len(observation_color), ncols=len(magnitude_edge) - 1, figure=figure, hspace=0.0, wspace=0.0)
    
    for i in range(len(magnitude_edge) - 1):
        for j in range(len(observation_color)):
            plot = figure.add_subplot(gridspec[j, i])
            
            observation_select = (magnitude_edge[i] < observation_dataset['mag_i_lsst']) & (observation_dataset['mag_i_lsst'] < magnitude_edge[i + 1])
            simulation_select = (magnitude_edge[i] < simulation_dataset['mag_i_lsst']) & (simulation_dataset['mag_i_lsst'] < magnitude_edge[i + 1])
            
            plot.scatter(simulation_redshift[simulation_select], simulation_color[j][simulation_select], s=0.05, c='darkgreen', marker='o', label=r'$\mathtt{CosmoDC2}$', alpha=0.05)
            plot.scatter(observation_redshift[observation_select], observation_color[j][observation_select], s=0.05, c='darkorange', marker='o', label=r'$\mathtt{OpenUniverse2024}$', alpha=0.05)
            
            plot.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
            plot.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
            
            plot.set_xlim(redshift1, redshift2)
            plot.set_ylim(color1[j], color2[j])
            plot.set_rasterized(True)
            
            if i == 0:
                plot.set_ylabel(label_list[j])
                
            else:
                plot.set_yticklabels([])
            
            if j == len(observation_color) - 1:
                plot.set_xlabel(r'$z_\mathrm{true}$')
                
            else:
                plot.set_xticklabels([])
            
            if j == 0:
                plot.set_title(r'${:.0f}'.format(magnitude_edge[i]) + r'\, \mathrm{mag} < i < ' + r'{:.0f}'.format(magnitude_edge[i + 1]) + r'\, \mathrm{mag}$')
            
            if j == 0 and i == 0:
                plot.text(1.2, 2.0, r'$\mathtt{CosmoDC2}$', color='darkgreen')
                plot.text(1.2, 1.5, r'$\mathtt{OpenUniverse2024}$', fontsize=20, color='darkorange')
    
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/CATALOG/'.format(tag)), exist_ok=True)
    
    figure.savefig(os.path.join(figure_folder, '{}/CATALOG/FIGURE.pdf'.format(tag)), format='pdf', bbox_inches='tight', dpi=256)
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Catalog')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)