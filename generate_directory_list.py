import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
import librosa

"""
Modified from: https://github.com/lesterphillip/torgo_vc

"""


def check_utt_length(file_path):
    """

    :param file_path:
    :return:
    """
    main_path = file_path.rsplit("/", 2)[0]
    file_name = file_path.split("/")[-1].split(".")[0]

    wav_location = f"{main_path}/wav_arrayMic/{file_name}.wav"
    try:
        y, sr = librosa.load(wav_location, 16000)
    except FileNotFoundError as e:
        wav_location = f"{main_path}/wav_headMic/{file_name}.wav"
        y, sr = librosa.load(wav_location, 16000)

    duration = librosa.get_duration(y=y, sr=sr)

    return duration, wav_location


def define_utt_type(transcript_prompt):
    """
    Classify utterance between a word, blabber or sentence.
    :param transcript_prompt: transcript of speech
    :return: type of speech
    """
    if " " not in transcript_prompt:
        return "word"

    elif "[" in transcript_prompt or "]" in transcript_prompt:
        return "blabber"

    else:
        return "sentence"

def find_data_postions(file_path):
    """
    Finds where the stars (location where speaker ID and general ID are in the dataset)
    :param file_path:
    :return:
    """
    file_path = np.array(file_path)

    elements = np.where(file_path == '*')[0]

    print(elements)
    location_dict = {
        "general_id" : elements[0],
        "speaker_id" : elements[1],
    }

    return location_dict


def check_transcripts(file_path):
    """

    :param file_path: Parent file path of the speech
    :return: Nothing returned. transcripts.csv file generated
    """

    all_files = glob.glob(file_path, recursive=True)

    file_path = file_path.replace("\\", "/")
    location_dict = find_data_postions(file_path.split("/"))

    total_actions = 0
    total_words = 0
    total_sents = 0

    general_ids = []
    spkr_ids = []
    transcripts = []
    directories = []
    utt_type = []
    file_duration = []

    print("Analyzing files...")

    for og_file in tqdm(all_files):
        file_ = og_file.replace("\\", "/")

        f_ = open(file_, "r")

        transcript_prompt = f_.read()
        transcript_prompt = transcript_prompt.strip("\n")

        if (transcript_prompt.split("/")[0] == "input"):
            continue
        else:
            transcripts.append(transcript_prompt)

        general_id = file_.split("/")[location_dict["general_id"]]
        general_ids.append(general_id)

        spkr_id = file_.split("/")[location_dict["speaker_id"]]
        spkr_ids.append(spkr_id)

        utt_type.append(define_utt_type(transcript_prompt))

        try:
            duration, wav_location = check_utt_length(file_)
            file_duration.append(duration)
            directories.append(wav_location)

        except FileNotFoundError as e:
            file_duration.append(np.NaN)
            directories.append(np.NaN)

    df = pd.DataFrame({
        "general_ids": general_ids,
        "speaker_ids": spkr_ids,
        "directory": directories,
        "transcripts": transcripts,
        "utt_type": utt_type,
        "duration": file_duration
    })

    df = df[df["directory"].notna()]
    df.to_csv(f"transcripts.csv", index=False)
    print(df.head())




def generate_directory_uaspeech(audio_file_path: str, transcript_file_path: str):
    """

    Args:
        audio_file_path: "./data/UASpeech/audio/*/*/*.wav"

    Returns:

    """

    # ./data/UASpeech/audio/original/M07/M07_B3_CW88_M7.wav
    all_audio_files = glob.glob(audio_file_path, recursive=True)

    try:
        xls = pd.ExcelFile(transcript_file_path)
        transcript_file_paths = pd.read_excel(xls, "Word_filename")
        transcript_file_paths['word_index'] = range(1, len(transcript_file_paths) + 1)
    except FileNotFoundError as e:
        raise FileNotFoundError("Excel Transcript file not found")

    general_ids = []
    spkr_ids = []
    word_ids = []
    mic = []
    directories = []

    file_duration = []

    for audio_file in tqdm(all_audio_files):
        audio_file_path = audio_file.replace("\\", "/")

        # print(file_path)
        split_path = audio_file_path.split("/")
        context = split_path[-1].split("_")

        if context[0][0].lower() == "c":
            if context[0][1].lower() == "m":
                general_ids.append("MC")
            else:
                general_ids.append("FC")
        else:
            general_ids.append(context[0][0])

        spkr_ids.append(context[0])
        if context[2][0:2].lower() == "uw":
            word_ids.append(context[1] + "_" + context[2])
        else:
            word_ids.append(context[2])
        mic.append(context[3].split(".")[0])

        try:
            directories.append(os.path.abspath(audio_file_path))

            # TODO Swap out commented code to turn file duration getting back on.

            # y, sr = librosa.load(audio_file_path, 16000)
            # duration = librosa.get_duration(y=y, sr=sr)
            # file_duration.append(duration)
            file_duration.append(np.NaN)

        except FileNotFoundError as e:
            file_duration.append(np.NaN)
            directories.append(np.NaN)

    df = pd.DataFrame({
        "general_ids": general_ids,
        "speaker_ids": spkr_ids,
        "directory": directories,
        "word_ids": word_ids,
        "mic": mic,
        "duration": file_duration
    })

    df = merge_uaspeech_audio_transcripts(df, transcript_file_paths)

    df.to_csv("UAspeech_transcripts.csv", index=False)

    return df


def merge_uaspeech_audio_transcripts(df: pd.DataFrame, transcript_file_paths: pd.DataFrame):
    output = pd.merge(df, transcript_file_paths, left_on="word_ids", right_on="FILE NAME", how="left")
    output = output.drop(columns=["FILE NAME"])

    output = output.rename(columns={"WORD": "transcripts"})
    return output

#
# if __name__ == "__main__":
# #     check_transcripts("./*/*0*/Session*/prompts/*.txt")
#     print("Hello World")
#     generate_directory_uaspeech("./data/UASpeech/audio/original/*/*.wav")