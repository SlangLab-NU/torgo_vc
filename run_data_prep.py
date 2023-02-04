import os

import pandas as pd
import yaml
import glob
import torch
import random
import argparse
import logging
import torchaudio
import numpy as np

from generate_directory_list import generate_directory_uaspeech
from rename_datasets import match_speakers

def get_data_prep_args():
    parser = argparse.ArgumentParser(description="Prepares UASpeech for VS or ASR")

    # "./data/UASpeech/audio/original/*/*.wav"
    parser.add_argument("-p", "--file_path", help="File path to UASpeech. ex: /data/UASpeech/audio/original/*/*.wav")
    parser.add_argument("-x", "--excel_file_path", help="File path to UAspeech excel mapping word ID to transcript")

    parser.add_argument("-r", "--create_transcript", help="--regenerate UAspeech transcript file",choices=["Y", "N", "y", "n"], default="N")
    parser.add_argument("-t", "--target_speaker", help="Target speaker to generate mapping for", required=True)
    parser.add_argument("-s", "--source_speaker", help="Source speaker that we are mapping target speaker to" )
    parser.add_argument("-random_seed", "--random_seed", help="Random seed for train test split", type=int, default=1)

    args = parser.parse_args()

    return args



def main():
    args = get_data_prep_args()
    if not os.path.isfile("UAspeech_transcripts.csv") or args.create_transcript.lower() == "y":
        if args.file_path == None or args.excel_file_path == None:
            raise FileNotFoundError("Path to audio file and path to excel file needed")
        else:
            generate_directory_uaspeech(args.file_path, args.excel_file_path)

    df = pd.read_csv("UAspeech_transcripts.csv")
    target_speaker = str(args.target_speaker).strip()

    if args.source_speaker == None:
        match_speakers(target_speaker, df)
    else:
        match_speakers(target_speaker,df=df , random_seed=args.random_seed, src_speaker=args.source_speaker)


    print(args.file_path)

if __name__ =="__main__":
    main()
