import torch, torchaudio
import IPython.display as ipd
import os
from argparse import ArgumentParser
from datasets import load_dataset, Audio
from copy import copy

# Get the audio paths
file_path_column = "filename_old"


def get_args():
    parser = ArgumentParser()
    parser.add_argument("dataset", nargs="*", default=("test"))
    parser.add_argument("--utterance-phrase", default="comb")
    parser.add_argument("--input-dir", default="/home/data1/psst-data/psst-data-2022-03-02-full")
    parser.add_argument("--output-dir", default="/home/data1/psst-data-out/KNN_VC_wav_files")
    return parser.parse_args()


def change_file_paths(data_instance, data_dir: str):
    """
    Updates the file path to retrieve the audio files for utterances
    :param data_instance: The test, train or valid dataset
    :param data_dir: The directory to join
    :return: The dataset with the updated directory path
    """
    data_instance[file_path_column] = os.path.join(data_dir,
                                                   data_instance[file_path_column])
    return data_instance


def collect_utterance_phrases(dataset, phrase):
    """
    Collects all instances of phrases being uttered by various speakers in the PSST
    dataset.
    :param dataset: The test, train or valid sets from the PSST data
    :param phrase: The phrase to do voice conversion on
    :return: a list of all utterances of a given phrase
    """
    phrase_list = []
    for i in range(len(dataset)):
        if phrase in dataset["utterance_id"][i]:
            phrase_list.append(dataset[i])
    if len(phrase_list) < 1:
        raise ValueError("Phrase does not exist in dataset")

    return phrase_list


def get_speaker_src(utterance_list, index):
    """
    Gets the source speaker and returns the path to the audio utterance
    :param utterance_list: The list of utterance for a given phrase
    :param index: The index position of the source speaker
    :return: The audio path of the source speaker
     """
    if index < len(utterance_list):
        speaker = utterance_list[index][file_path_column]
        return speaker["path"]
    else:
        raise ValueError("Index is out of range of speaker list")


def get_reference_speakers(utterance_list, index):
    """
    Generates a list of speakers for a given phrase to use as reference for voice conversion
    :param utterance_list: The list of utterance for a given phrase
    :param index: The index position of the source speaker
    :return: The list of speaker audio paths to use as reference for voice conversion
     """
    # temp_list = copy(utterance_list)
    utterance_wav_paths = []
    print("{} : {}".format(index, len(utterance_list)))
    if index < len(utterance_list):
        for i in range(len(utterance_list)):
            if i == index:
                continue
            else:
                utterance = utterance_list[i][file_path_column]["path"]
                utterance_wav_paths.append(utterance)
        return utterance_wav_paths
    else:
        raise ValueError("Index is out of range of speaker list")


def generate_vc(dataset, phrase, output_dir):
    """
    Generates a list of speakers for a given phrase to use as reference for voice conversion
    :param utterance_list: The list of utterance for a given phrase
    :param index: The index position of the source speaker
    :return: The list of speaker audio paths to use as reference for voice conversion
     """
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device='cuda')
    utterances = collect_utterance_phrases(dataset, phrase)

    for i in range(len(utterances) - 1):
        src_path = get_speaker_src(utterances, i)
        reference_wav_paths = get_reference_speakers(utterances, i)
        # path to 16kHz, single-channel, source waveform
        src_wav_path = src_path
        # list of paths to all reference waveforms (each must be 16kHz, single-channel) from the target speaker
        ref_wav_paths = reference_wav_paths
        query_seq = knn_vc.get_features(src_wav_path)
        matching_set = knn_vc.get_matching_set(ref_wav_paths)

        out_wav = knn_vc.match(query_seq, matching_set, topk=4)

        ipd.Audio(out_wav.numpy(), rate=16000)

        save_dir = "{}/{}_out_V{}.wav".format(output_dir, phrase, i)
        torchaudio.save(save_dir, out_wav[None], 16000)


def main(dataset,
         utterance_phrase: str,   
         input_dir: str, 
         output_dir: str):
    dataset_dict = load_dataset('csv', data_files={
        "train": '/home/data1/psst-data-csv/train_utterances_excel.csv',
        "valid": '/home/data1/psst-data-csv/valid_utterances_excel.csv',
        "test": '/home/data1/psst-data-csv/test_utterances_excel.csv'
    })

    for dataset in ("train", "valid", "test"):
        data_dir = os.path.join(input_dir, dataset)
        dataset_dict[dataset] = dataset_dict[dataset].map(change_file_paths,
                                                          fn_kwargs={"data_dir": data_dir})
        dataset_dict[dataset] = dataset_dict[dataset].cast_column(file_path_column,
                                                                  Audio(sampling_rate=16000))

    generate_vc(dataset_dict[dataset], utterance_phrase, output_dir)


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
    data_input_dir = "/home/data1/psst-data/psst-data-2022-03-02-full"
    data_output_dir = "/home/data1/psst-data-out/KNN_VC_wav_files"
