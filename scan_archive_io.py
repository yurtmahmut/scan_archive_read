import sys
import time
import argparse
import os

import numpy as np

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "orchestra"))

from GERecon import Archive

def get_kspace(archive):
    
    ksp = []
    try:
        while True:
            frame = np.array(archive.NextFrame()[..., None], dtype=np.complex64)
            ksp.append(frame)
    
    except Exception:
        pass

    if len(ksp) > 1 and ksp[0].shape != ksp[1].shape:
        ksp.pop(0)
    
    if ksp:
        kspace = np.concatenate(ksp, axis=-1)
        return kspace
    else:
        raise ValueError("No valid frames were found in the archive.")


def main(args):

    print(">>> Scan archive loading... ", end="", flush=True)
    start_time = time.perf_counter()
    archive = Archive(args.inp_folder)
    elapsed_time = time.perf_counter() - start_time
    print(f"done. Time taken: {elapsed_time:.2f} seconds.", flush=True)

    print(">>> Raw k-space data retrieving... ", end="", flush=True)
    start_time = time.perf_counter()
    ksp = get_kspace(archive)
    elapsed_time = time.perf_counter() - start_time
    print(f"done. Time taken: {elapsed_time:.2f} seconds.", flush=True)

    print(">>> Saving k-space data as .npy file... ", end="", flush=True)
    start_time = time.perf_counter()
    np.save(os.path.join(args.out_folder, f"ksp_{args.out_filename}"), ksp)
    elapsed_time = time.perf_counter() - start_time
    print(f"done. Time taken: {elapsed_time:.2f} seconds.", flush=True)


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Scan archive to .npy for GE scanner MRI data (readout, coils, frames).")
    parser.add_argument('--inp_folder', type=str, required=True, help="Folder to scan archive.")
    parser.add_argument('--out_folder', type=str, required=True, help="Folder to save file.")
    parser.add_argument('--out_filename', type=str, required=True, help="Filename to save file.")
    
    return parser


if __name__ == '__main__':

    start_time = time.perf_counter()
    args = create_arg_parser().parse_args(sys.argv[1:])

    file_list = [os.path.join(folder, file)
                 for folder, subfolders, files in os.walk(os.path.abspath(args.inp_folder))
                 for file in files]
    
    largest_file = max(file_list, key=os.path.getsize)
    args.inp_folder = largest_file
    
    main(args)
    elapsed_time = time.perf_counter() - start_time
    print(f">>> Total time: {elapsed_time:.2f} seconds.")
