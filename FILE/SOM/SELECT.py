import os
import time
import h5py
import numpy
import argparse
from rail import core


def main(index, folder):
    '''
    Summarize the true redshift distribution of the lens and source samples.
    
    Arguments:
        index (int): The index of the dataset.
        folder (str): The path to the base folder.
    
    Returns:
        duration (float): The time taken to summarize the redshift distribution.
    '''
    # Start
    start = time.time()
    print('Index:{}'.format(index))
    
    # Path
    som_folder = os.path.join(folder, 'SOM')
    
    # Bin
    with h5py.File(os.path.join(som_folder, 'LENS/LENS{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_lens = file['bin'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(som_folder, 'SOURCE/SOURCE{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_source = file['bin'][:].astype(numpy.float32)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_delta = (z2 - z1) / grid_size
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    z_data = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, grid_size)
    
    # Load
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    width = 1000
    
    # Lens
    for m in range(1, len(bin_lens)):
        # Select
        cell_name = os.path.join(som_folder, 'LENS/LENS{}/CELL{}.hdf5'.format(index, m))
        sample_name = os.path.join(som_folder, 'LENS/LENS{}/SAMPLE{}.hdf5'.format(index, m))
        cluster_name = os.path.join(som_folder, 'LENS/LENS{}/CLUSTER{}.hdf5'.format(index, m))
        
        cell_data = data_store.read_file(key='cell', path=cell_name, handle_class=core.data.TableHandle)()
        sample_data = data_store.read_file(key='sample', path=sample_name, handle_class=core.data.TableHandle)()
        cluster_data = data_store.read_file(key='cluster', path=cluster_name, handle_class=core.data.TableHandle)()
        
        cell_id = cell_data['cluster_ind'][:].astype(numpy.int32)
        cluster_id = cluster_data['uncovered_clusters'][:].astype(numpy.int32)
        z_true = sample_data['photometry']['redshift'][:].astype(numpy.float32)
        
        # Single
        lens_single = numpy.zeros(grid_size + 1)
        lens_sample = numpy.zeros((width, grid_size + 1))
        z_select = z_true[~numpy.isin(cell_id, cluster_id)]
        
        histogram = numpy.histogram(z_select, bins=z_grid, density=True, range=(z1, z2))[0]
        single = numpy.interp(x=z_grid, xp=z_data, fp=histogram, left=0.0, right=0.0)
        lens_single = single / single.sum() / z_delta
        
        # Sample
        for n in range(width):
            z_sample = numpy.random.choice(z_select, size=(width, z_select.size), replace=True)
            
            histogram = numpy.histogram(z_sample[n, :], bins=z_grid, density=True, range=(z1, z2))[0]
            sample = numpy.interp(x=z_grid, xp=z_data, fp=histogram, left=0.0, right=0.0)
            lens_sample[n, :] = sample / sample.sum() / z_delta
        
        # Save
        lens_data = {'single': lens_single, 'sample': lens_sample}
        with h5py.File(os.path.join(som_folder, 'LENS/LENS{}/SELECT{}.hdf5'.format(index, m)), 'w') as file:
            for key, value in lens_data.items():
                file.create_dataset(key, data=value)
    
    # Source
    for m in range(1, len(bin_source)):
        # Select
        cell_name = os.path.join(som_folder, 'SOURCE/SOURCE{}/CELL{}.hdf5'.format(index, m))
        sample_name = os.path.join(som_folder, 'SOURCE/SOURCE{}/SAMPLE{}.hdf5'.format(index, m))
        cluster_name = os.path.join(som_folder, 'SOURCE/SOURCE{}/CLUSTER{}.hdf5'.format(index, m))
        
        cell_data = data_store.read_file(key='cell', path=cell_name, handle_class=core.data.TableHandle)()
        sample_data = data_store.read_file(key='sample', path=sample_name, handle_class=core.data.TableHandle)()
        cluster_data = data_store.read_file(key='cluster', path=cluster_name, handle_class=core.data.TableHandle)()
        
        cell_id = cell_data['cluster_ind'][:].astype(numpy.int32)
        cluster_id = cluster_data['uncovered_clusters'][:].astype(numpy.int32)
        z_true = sample_data['photometry']['redshift'][:].astype(numpy.float32)
        
        # Single
        source_single = numpy.zeros(grid_size + 1)
        source_sample = numpy.zeros((width, grid_size + 1))
        z_select = z_true[~numpy.isin(cell_id, cluster_id)]
        
        histogram = numpy.histogram(z_select, bins=z_grid, density=True, range=(z1, z2))[0]
        single = numpy.interp(x=z_grid, xp=z_data, fp=histogram, left=0.0, right=0.0)
        source_single = single / single.sum() / z_delta
        
        # Sample
        for n in range(width):
            z_sample = numpy.random.choice(z_select, size=(width, z_select.size), replace=True)
            
            histogram = numpy.histogram(z_sample[n, :], bins=z_grid, density=True, range=(z1, z2))[0]
            sample = numpy.interp(x=z_grid, xp=z_data, fp=histogram, left=0.0, right=0.0)
            source_sample[n, :] = sample / sample.sum() / z_delta
        
        # Save
        source_data = {'single': source_single, 'sample': source_sample}
        with h5py.File(os.path.join(som_folder, 'SOURCE/SOURCE{}/SELECT{}.hdf5'.format(index, m)), 'w') as file:
            for key, value in source_data.items():
                file.create_dataset(key, data=value)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='SOM Select')
    PARSE.add_argument('--index', type=int, help='The index of the dataset')
    PARSE.add_argument('--folder', type=str, help='The path to the base folder')
    
    # Parse
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(INDEX, FOLDER)