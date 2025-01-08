
import os
import time
import h5py
import numpy
import argparse
from rail import core


def main(index, folder):
    '''
    Generate the datasets for the SOM summarizer.
    
    Arguments:
        index (int): The number of datasets.
        folder (str): The base folder of the datasets.
    
    Returns:
        float: The duration of the plotter.
    '''
    # Start
    start = time.time()
    print('Index:{}'.format(index))
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    som_folder = os.path.join(folder, 'SOM/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    os.makedirs(os.path.join(som_folder, 'LENS'), exist_ok=True)
    os.makedirs(os.path.join(som_folder, 'LENS/LENS{}'.format(index)), exist_ok=True)
    
    os.makedirs(os.path.join(som_folder, 'SOURCE'), exist_ok=True)
    os.makedirs(os.path.join(som_folder, 'SOURCE/SOURCE{}'.format(index)), exist_ok=True)
    
    # Bin
    with h5py.File(os.path.join(fzb_folder, 'LENS/LENS{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_lens = file['bin'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(fzb_folder, 'SOURCE/SOURCE{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_source = file['bin'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(som_folder, 'LENS/LENS{}/BIN.hdf5'.format(index)), 'w') as file:
        file.create_dataset('bin', data=bin_lens)
    
    with h5py.File(os.path.join(som_folder, 'SOURCE/SOURCE{}/BIN.hdf5'.format(index)), 'w') as file:
        file.create_dataset('bin', data=bin_source)
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    
    z1_source = 0.0
    z2_source = 3.0
    
    grid_size = 300
    z_grid = numpy.linspace(z1_source, z2_source, grid_size + 1)
    
    # Load
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    application_name = os.path.join(dataset_folder, 'APPLICATION/DATA{}.hdf5'.format(index))
    application_dataset = data_store.read_file(key='application', path=application_name, handle_class=core.data.TableHandle)()
    
    mag_source = application_dataset['photometry']['mag_i_lsst']
    
    estimate_name = os.path.join(fzb_folder, 'ESTIMATE/ESTIMATE{}.hdf5'.format(index))
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)()
    
    z_mean = numpy.concatenate(estimator.mean())
    z_median = numpy.concatenate(estimator.median())
    z_mode = numpy.concatenate(estimator.mode(z_grid))
    z_phot = numpy.average([z_mean, z_median, z_mode], axis=0)
    del z_mean, z_median, z_mode, estimator
    
    # Save Select
    slope = 4.0
    intercept = 18.0
    select_source = numpy.isfinite(z_phot) & (z1_source < z_phot) & (z_phot < z2_source)
    select_lens = numpy.isfinite(z_phot) & (z1_lens < z_phot) & (z_phot < z2_lens) & (mag_source < slope * z_phot + intercept)
    
    # Save Lens
    for m in range(1, len(bin_lens)):
        select_lens_bin = select_lens & (bin_lens[m - 1] < z_phot) & (z_phot < bin_lens[m])
        
        with h5py.File(os.path.join(som_folder, 'LENS/LENS{}/SAMPLE{}.hdf5'.format(index, m)), 'w') as file: 
            for name in application_dataset.keys():
                file.create_group(name=name)
                for key, value in application_dataset[name].items():
                    file[name].create_dataset(key, data=value[select_lens_bin])
    
    # Save Source
    for m in range(1, len(bin_source)):
        select_source_bin = select_source & (bin_source[m - 1] < z_phot) & (z_phot < bin_source[m])
        
        with h5py.File(os.path.join(som_folder, 'SOURCE/SOURCE{}/SAMPLE{}.hdf5'.format(index, m)), 'w') as file: 
            for name in application_dataset.keys():
                file.create_group(name=name)
                for key, value in application_dataset[name].items():
                    file[name].create_dataset(key, data=value[select_source_bin])
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='SOM Figure')
    PARSE.add_argument('--index', type=int, help='The index of the dataset')
    PARSE.add_argument('--folder', type=str, help='The path to the base folder')
    
    # Parse
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(INDEX, FOLDER)