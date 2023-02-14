import os

import pandas as pd
import argparse
import yaml

from generate_directory_list import generate_directory_uaspeech
from rename_datasets import match_speakers, save_split_df

def get_data_prep_args():
    parser = argparse.ArgumentParser(description="Prepares UASpeech for VS or ASR")

    # "./data/UASpeech/audio/original/*/*.wav"
    # parser.add_argument("-p", "--file_path", help="File path to UASpeech. ex: /data/UASpeech/audio/original/*/*.wav")
    # parser.add_argument("-x", "--excel_file_path", help="File path to UAspeech excel mapping word ID to transcript")

    parser.add_argument("-r", "--create_transcript", help="--regenerate UAspeech transcript file",choices=["Y", "N", "y", "n"], default="N")
    parser.add_argument("-t", "--target_speaker", help="Target speaker to generate mapping for", required=True)
    parser.add_argument("-s", "--source_speaker", help="Source speaker that we are mapping target speaker to" )
    parser.add_argument("-random_seed", "--random_seed", help="Random seed for train test split", type=int, default=1)

    args = parser.parse_args()

    return args



def main():
    args = get_data_prep_args()

    if not os.path.isfile("UAspeech_transcripts.csv") or args.create_transcript.lower() == "y":
        with open('config.yaml', 'r') as f:
            doc = yaml.load(f)

        excel_file_path = doc["file_paths"]["excel_file_path"]
        file_path = doc["file_paths"]["data_file_path"]
        if file_path is None or excel_file_path is None:
            raise FileNotFoundError("Path to audio file and path to excel file needed")
        else:
            generate_directory_uaspeech(file_path, excel_file_path)

    df = pd.read_csv("UAspeech_transcripts.csv")
    target_speaker = str(args.target_speaker).strip()
    if args.source_speaker is None:
        train, dev, test = match_speakers(target_speaker, df,
                                          random_seed=args.random_seed)
    else:
        train, dev, test = match_speakers(target_speaker,
                                          df=df ,
                                          random_seed=args.random_seed,
                                          src_speaker=args.source_speaker)

    save_split_df(train, "train")
    save_split_df(dev, "dev")
    save_split_df(test, "test")

if __name__ =="__main__":
    main()
