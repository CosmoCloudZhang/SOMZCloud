import os
import json
import time
import numpy
import argparse


def main(folder):
    '''
    Store the fiducial values of survey configuration
    
    Arguments:
        folder (str): The base folder of the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    info_folder = os.path.join(folder, 'INFO/')
    
    # Survey
    survey = {
        'Y1': {
            'AREA': 18000, 
        }, 
        'Y10': {
            'AREA': 18000,
        }
    }
    
    # Loop
    sky = 4 * numpy.pi * numpy.square(180 / numpy.pi)
    for tag in survey.keys():
        print('Tag: {}'.format(tag))
        survey[tag]['FRACTION'] = survey[tag]['AREA'] / sky
    
    # Save
    with open(os.path.join(info_folder, 'SURVEY.json'), 'w') as file:
        json.dump(survey, file, indent=4)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info Survey')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(FOLDER)