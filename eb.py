import os
import csv
import argparse
import numpy as np

from glob import glob
from pyplanter import PlantedH5


# output format specific constants
CSV_SEP = ','
DATACACHE = 'RAW'
HEADER_SIZE = 6

OUTPUT_SETTINGS = {
    'valid_cols': (),
    'channel_names': (),
    'units': (),
    'multiplier': (),
}


def read_header(file_path:str):
    """
    Reads first 5 rows of .out file

    input: file_path -> str: absolute path to .out file
    output: header -> dict: header from .out file
    """

    header = dict()

    # open .out file and read header
    with open(file_path, 'r') as fh:
        for row in fh:
            if row[:2] != '# ':
                break
            param_name, param_val = row.lstrip('# ').rstrip('\n').split(': ')

            if param_name == 'sampleFrequency':
                param_val = int(param_val.split('.')[0])
            elif param_name == 'fancyNames':
                param_val = param_val.replace('" "', ';').replace('"', '').split(';')
            else:
                param_val = param_val.split(' ')

            header[param_name] = param_val


    # filter out and sort columns as defined by custom config
    if OUTPUT_SETTINGS['valid_cols']:
        header['columns'] = OUTPUT_SETTINGS['channel_names'] if OUTPUT_SETTINGS['channel_names'] else [header['columns'][idx] for idx in OUTPUT_SETTINGS['valid_cols']]
        header['units'] = OUTPUT_SETTINGS['units'] if OUTPUT_SETTINGS['units'] else [header['units'][idx] for idx in OUTPUT_SETTINGS['valid_cols']]
        header['fancyNames'] = [header['fancyNames'][idx] for idx in OUTPUT_SETTINGS['valid_cols']]
        header['calibrationSlots'] = [header['calibrationSlots'][idx] for idx in OUTPUT_SETTINGS['valid_cols']]
        
    return header


def to_h5(file_path:str, export_path:str, file_name:str, chunk_size:int=None, header:dict=None):
    """Converts data stored in .out file into h5 file

    Args:
        file_path (str): _description_
        export_path (str): _description_
        file_name (str): _description_
        chunk_size (int, optional): _description_. Defaults to None.
        header (str, optional): _description_. Defaults to None.
    """
    
    # read .out file
    print('...exporting data into h5 file...', end ="", flush=True)

    # create new h5 file
    planter = PlantedH5()    
    planter.create(os.path.join(export_path, file_name), sampl_freq=header['sampleFrequency'])

    data_chunk = []
    with open(file_path, 'r') as fp:            

        # skip first n rows (header)
        for _ in range(HEADER_SIZE):
            next(fp)

        reader = csv.reader(fp, delimiter=' ')
        data_chunk = np.array(list(reader)).astype(float)
    
    if OUTPUT_SETTINGS['valid_cols']:
        data_chunk = data_chunk[:, OUTPUT_SETTINGS['valid_cols']]

    data_chunk = np.transpose(data_chunk)

    if OUTPUT_SETTINGS['multiplier']:
        coeffs = np.broadcast_to(
            np.array(OUTPUT_SETTINGS['multiplier'])[:, np.newaxis],
            data_chunk.shape,
            )
        data_chunk = np.multiply(data_chunk, coeffs)

    # create new dataset
    planter.create_dataset(
        data_chunk,
        ch_names=header['columns'],
        datacache_name=DATACACHE,
        unit_name=header['units'],
        )

    planter.flush()

    planter.close()

    print('DONE')


def to_csv(file_path:str=None, export_path:str=None, file_name:str=None, chunk_size:int=None, header:str=None):
    """Converts data stored in .out file into csv file

    input: file_path -> str: absolute path to .out file
    input: export_path -> str: absolute path to export folder. Folder must exist before export.
    input: file_name -> str: name of file without suffix
    input: skip_rows -> int: number of header rows to skip (default=6)
    input: chunk_size -> int: TO DO
    output: None
    """

    file_name += '.csv'

    print('...exporting data into csv file...', end ="", flush=True)
    with open(file_path, 'r') as fp:
        with open(os.path.join(export_path, file_name), 'w') as ep:
            # write header
            if header:
                ep.write(header+'\n')

            # skip first n rows (header)
            for _ in range(HEADER_SIZE):
                next(fp)

            for row in fp:
                data_chunk = row.rstrip('\n').split(' ')
                data_chunk = ''.join([data_chunk[idx]+CSV_SEP for idx in OUTPUT_SETTINGS['valid_cols']])[:-1]                

                # write data chunk
                ep.write(data_chunk+'\n')

    print('DONE')


def to_sel(header:dict, export_path:str, file_name:str):
    """
    Creates .sel file from header.

    input: header -> dict: header from .out file parsed by read_header().    
    input: export_path -> str: absolute path to export folder. Folder must exist before export.
    input: file_name -> str: name of file without suffix
    output: None
    """

    columns_txt = ''.join(['%'+item+'\t1\n' for item in header['columns']])    

    print('creating .sel file...', end ="", flush=True)
    with open(os.path.join(export_path, file_name + '.sel'), 'w') as sf:
        sf.writelines(
            '%SignalPlant ver.:1.2.7.3\n'
            '%Selection export from file:\n'
            f'%{file_name + ".csv"}\n'
            f'%SAMPLING_FREQ [Hz]:{header["sampleFrequency"]}\n'
            '%CHANNELS_VALIDITY-----------------------\n'
            f'{columns_txt}'
            '%----------------------------------------\n'
            '%Structure:\n'
            '%Index[-], Start[sample], End[sample], Group[-], Validity[-], Channel Index[-], Channel name[string], Info[string]\n'
            '%Divided by: ASCII char no. 9\n'
            '%DATA------------------------------------\n'
            f'1\t1\t2\t0\t0.00000\t1\t%time\t1\n'
        )    

    print('DONE', end =" ", flush=True)    


def run(input_folder_path:str, output_folder_path:str, file_name:str=None, output_format:str='h5'):        
    
    assert output_format in {'h5', 'csv'}

    # assing general *. file names to perform batch processing 
    if not file_name:
        file_name = '*.out'
    
    # list required files in input folder    
    file_list = glob(os.path.join(input_folder_path, f'{file_name}'))
    
    # return if not found any .out file
    if not file_list:
        print(f'InputError: no file(s) were found in "{input_folder_path}". Check if the path or file name are correct.')
        return

    # make output directory
    dir_name = os.path.basename(input_folder_path)
    export_path = os.path.join(output_folder_path, dir_name)

    try:
        os.makedirs(export_path)
    except OSError as e:
        pass            

    # process files
    for file_path in file_list:        
        file_name = os.path.basename(file_path)[:-4]
        print(f'Converting file {file_name}...', end=" ", flush=True)
        
        # rename output file to avoid Signal Plant loading issues        
        file_name = file_name.replace(' ', '_')
        file_name = file_name.replace('-', '_')
        file_name = file_name.replace('.', '-')

        # read header and convert to .sel file
        header = read_header(file_path)
        
        if output_format == 'csv':
            # export *.sel file        
            to_sel(
                header,
                export_path,
                file_name,            
                )        

            # export data
            to_csv(
                file_path,
                export_path,
                file_name,
                header=f'{CSV_SEP} '.join(header['columns']),
                )
        
        if output_format == 'h5':
            to_h5(
                file_path,
                export_path,
                file_name,
                header=header,
            )
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='', help='A path to folder with .out files.')
    parser.add_argument('-o', '--output', type=str, default='', help='A path to export folder.')
    parser.add_argument('-f', '--fileName', type=str, default='', help='Name of file to export.')
    parser.add_argument('-oo', '--outputFormat', type=str, default='h5', help='Format of output file.')

    args = parser.parse_args()

    run(
        input_folder_path=args.input,
        output_folder_path=args.output,
        file_name=args.fileName,
        output_format=args.outputFormat,
    )