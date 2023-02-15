import numpy as np
import pandas
import pandas as pd
from tqdm import tqdm
import os
import glob
import librosa
import soundfile as sf
from math import floor, ceil
from sklearn.model_selection import train_test_split

"""
Modified from: https://github.com/lesterphillip/torgo_vc

"""


def open_and_save_wav(file_path, speaker_id, new_id, split_type):
    y, sr = librosa.load(file_path, sr = 16000)
    new_path = f"output/{split_type}/{speaker_id}/{new_id}.wav"
    make_directory = new_path.rsplit("/", 1)[0]

    if not os.path.exists(make_directory):
        os.makedirs(make_directory)

    sf.write(new_path, y, sr)


def save_split_df(df: pandas.DataFrame, split: str):

    df.apply(lambda row: open_and_save_wav(row['directory_x'], row['speaker_ids_x'], row["word_ids_x"], split), axis=1)

    df.apply(lambda row: open_and_save_wav(row["directory_y"], row['speaker_ids_y'], row["word_ids_y"], split), axis=1)


def process_csv_file(df_dys, df_nondys, gender):
    df_res = pd.merge(df_dys, df_nondys, on="transcripts")
    df_res = df_res.loc[df_res["transcripts"] != "[relax your mouth in its normal position]"]
    df_res = df_res.drop_duplicates(subset="directory_x")
    df_res = df_res.reset_index(drop=True)

    print(f"Dysarthric duration: {df_res['duration_x'].sum()}")
    print(f"Non-dysarthric duration: {df_res['duration_y'].sum()}")

    df_res.to_csv(f"paired_transcripts_{gender}.csv", index=False)

    for index, row in tqdm(df_res.iterrows(), total=len(df_res)):
        wav_f = row["directory_x"]
        wav_fc = row["directory_y"]
        transcript = row["transcripts"]

        if ".wav" not in str(wav_f) or ".wav" not in str(wav_fc):
            continue

        if index <= 0.6 * floor(len(df_res)):
            split_type = "train"
        elif index <= 0.8 * floor(len(df_res)) and index > 0.6 * floor(len(df_res)):
            split_type = "dev"
        elif index <= 0.9 * floor(len(df_res)) and index > 0.8 * floor(len(df_res)):
            split_type = "test1"
        else:
            split_type = "test2"

        new_id = str(index).zfill(4)
        # open_and_save_wav(wav_f, new_id, split_type)
        # open_and_save_wav(wav_fc, new_id, split_type)

        make_directory = "/transcripts"

    df_res.to_csv("total_summary.csv", index=False)


def match_speakers(trgspk: str, df: pd.DataFrame, random_seed: int = 1, src_speaker: str = "na", match_mic: str = "n"):
    """
    Function maps Torgo speech based on the transcript from either many to one or one to one
    Args:
        trgspk: Target speaker that the train/dev split is being made for
        df: Dataframe of all speech samples

        df: Dataset that is being split

        random_seed: Seed number for the train test split

        src_speaker: If the mapping is going to be one to one instead of many to one, include a src speaker.

    Returns: Train test split dataframe
    """

    if trgspk not in df["speaker_ids"].values:
        raise ValueError("Target speaker not in dataset")

    trgspk_df = df.loc[df["speaker_ids"] == trgspk]
    trgspk_df = trgspk_df.drop_duplicates(subset=["transcripts"])

    if src_speaker != "na":
        if src_speaker in df["speaker_ids"].values:
            df = df.loc[df["speaker_ids"] == src_speaker]
        else:
            raise ValueError("Source speaker not in dataset")

    else:
        df = df.loc[(df["general_ids"] != "MC") & (df["general_ids"] != "FC")]

    # Possible error if we are matching mics, but currently unsure and dataset might be too small regardless.
    # Might drop duplicate that might mic match later.

    df = df[df.mic != "M1"]

    df = df.drop_duplicates(subset=["speaker_ids", "transcripts"])




    if match_mic.lower() == "y":
        df = df.merge(trgspk_df, on=["transcripts", "mic"])

        # Element of randomness as mode can change if there is a tie.
        max_occurence_mic = df["mic"].mode().values[0]

        df = df[df.mic == max_occurence_mic]

    else:
        df = df.merge(trgspk_df, on=["transcripts"])


    output = df
    output.to_csv(f"{trgspk}_paired.csv", index=False)

    # TODO Train test split. How to split up the transcripts other than randomly
    train, test = train_test_split(output, test_size=0.2, random_state=random_seed)
    dev, test = train_test_split(test, test_size=0.5, random_state=random_seed)
    return train, test, dev

