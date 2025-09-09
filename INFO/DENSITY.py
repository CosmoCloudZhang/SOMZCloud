import os
import yaml
import h5py
import json
import time
import numpy
import argparse


def main(number, folder):
    '''
    Store the fiducial values of density configuration
    
    Arguments:
        number (int): The number of the datasets
        folder (str): The base folder of the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    
    # Path
    info_folder = os.path.join(folder, 'INFO/')
    model_folder = os.path.join(folder, 'MODEL/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Load
    with open(os.path.join(dataset_folder, 'CATALOG/OBSERVE.yaml'), 'r') as file:
        observe = yaml.safe_load(file)
        observation_list = observe['healpix_pixels']
        area = observe['sky_area'] / len(observation_list) * len(observation_list) // 2
    
    # Definition
    sigma0 = 0.26
    density = {'Y1': {}, 'Y10': {}}
    factor_list = [0.0, 0.5, 1.0, 2.0]
    label_list = ['ZERO', 'HALF', 'UNITY', 'DOUBLE']
    
    # Loop
    for tag in density.keys():
        print('Tag: {}'.format(tag))
        
        # Bin
        with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA0.hdf5'.format(tag)), 'r') as file:
            bin_lens_size = len(file['bin_lens'][...]) - 1
            bin_source_size = len(file['bin_source'][...]) - 1
        sample_size = 100
        
        # Density
        density_lens = numpy.zeros((bin_lens_size, number))
        density_source = numpy.zeros((bin_source_size, number))
        
        # Factor
        sigma_lens = numpy.zeros((number, bin_lens_size, sample_size))
        sigma_source = numpy.zeros((number, bin_source_size, sample_size))
        
        metric_lens = numpy.zeros((number, bin_lens_size, sample_size))
        metric_source = numpy.zeros((number, bin_source_size, sample_size))
        
        fraction_lens = numpy.zeros((number, bin_lens_size, sample_size))
        fraction_source = numpy.zeros((number, bin_source_size, sample_size))
        
        for n in range(number):
            print('Index: {}'.format(n + 1))
            
            with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/HISTOGRAM.hdf5'.format(tag, n + 1)), 'r') as file:
                sigma_lens[n, :, :] = file['sigma'][...]
                metric_lens[n, :, :] = file['metric'][...]
                fraction_lens[n, :, :] = file['fraction'][...]
            
            with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/HISTOGRAM.hdf5'.format(tag, n + 1)), 'r') as file:
                sigma_source[n, :, :] = file['sigma'][...]
                metric_source[n, :, :] = file['metric'][...]
                fraction_source[n, :, :] = file['fraction'][...]
            
            # Application
            with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, n + 1)), 'r') as file:
                application_sigma = file['morphology']['sigma'][...]
            
            # Lens
            with h5py.File(os.path.join(model_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, n + 1)), 'r') as file:
                select_lens = file['select'][...]
            
            for m in range(bin_lens_size):
                density_lens[m, n] = numpy.sum(select_lens[m, :]) / area / 3600
            
            # Source
            with h5py.File(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/SELECT.hdf5'.format(tag, n + 1)), 'r') as file:
                select_source = file['select'][...]
            
            for m in range(bin_source_size):
                density_source[m, n] = numpy.sum(numpy.square(sigma0) / (numpy.square(application_sigma[select_source[m, :]]) + numpy.square(sigma0))) / area / 3600
        
        factor_sigma_lens = numpy.sum(numpy.square(sigma_lens), axis=1)
        factor_metric_lens = numpy.sum(numpy.square(metric_lens), axis=1)
        factor_fraction_lens = numpy.square(numpy.sum(fraction_lens, axis=1))
        factor_lens = factor_fraction_lens / factor_sigma_lens / factor_metric_lens
        
        factor_sigma_source = numpy.prod(sigma_source, axis=1)
        factor_metric_source = numpy.prod(metric_source, axis=1)
        factor_fraction_source = numpy.prod(fraction_source, axis=1)
        factor_source = factor_fraction_source / factor_sigma_source / factor_metric_source
        
        # Loop
        for factor, label in zip(factor_list, label_list):
            print('Factor: {:.1f}, Label: {}'.format(factor, label))
            
            # Weight
            weight_lens = numpy.mean(numpy.power(factor_lens, factor), axis=1)
            weight_source = numpy.mean(numpy.power(factor_source, factor), axis=1)
            
            density[tag][label] = {}
            density[tag][label]['LENS'] = list(numpy.average(density_lens, weights=weight_lens, axis=1))
            density[tag][label]['SOURCE'] = list(numpy.average(density_source, weights=weight_source, axis=1))
    
    # Save
    with open(os.path.join(info_folder, 'DENSITY.json'), 'w') as file:
        json.dump(density, file, indent=4)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info density')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(NUMBER, FOLDER)