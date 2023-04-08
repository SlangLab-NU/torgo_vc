import pandas
import pandas as pd
import librosa
from tqdm import tqdm

def columns_to_txt(df: pandas.DataFrame, col_1_name: str, col_2_name: str):
    """
    Set two columns of a dataframe into text format with a space seperating them
    :param df: Datafram input
    :param col_1_name: Name of the first column
    :param col_2_name: Name of the second column
    :return: string with a space between the values of the two columns
    """
    str_output = ""
    for index, row in df.iterrows():
        str_output += str(row[col_1_name]) + " " + str(row[col_2_name]) + "\n"

    return str_output


def columns_to_data(df: pandas.DataFrame, col_1_name: str, col_2_name: str):
    """

    :param df: Dataframe
    :param col_1_name: Name of the first column
    :param col_2_name: Name of the second column
    :return: Outputs string in the form of the data file found in the arctic dataset for ESPNET voice conversion/asr
    """

    df = df.drop_duplicates(subset=["word_ids"])
    str_output = ""
    for index, row in df.iterrows():
        str_output += "( " + str(row[col_1_name]) + ' "' + str(row[col_2_name]) + '" ) \n'

    return str_output

    return str_output


def nemo_data_prep(df: pandas.DataFrame, file_path_col: str, text_col: str, length_col:str = "na"):
    """
    Takes a pandas DF and converts it into a json_str for the nvidia NEMO system.
    :param df:Dataframe that is inputted
    :param file_path_col: Column containing the file path
    :param text_col: Column containing the text transcriptions
    :param length_col: Column name for the length of the audio transcript
    :return: JSON string for input to the NEMO system.
    """

    json_str = ""

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        audio_filepath = '"audio_filepath": ' + '"' + str(row[file_path_col]) + '"'
        text = '"text": ' + '"' + str(row[text_col]) + '"'

        length_str = '"duration": '
        if length_col == "na":
            y, sr = librosa.load(str(row[file_path_col]))
            duration = librosa.get_duration(y=y, sr=sr)
            length_str += str(duration)
        else:
            length_str += str(row[length_col])

        json_str += "{" + audio_filepath + ", " + text + ", " + length_str + "}" + "\n"


    json_str = json_str.rstrip("\n")

    return json_str


def main():

    df = pd.read_csv("torgo_transcripts.csv")
    f = open("train.json", "w")

    nemo_data = nemo_data_prep(df=df, file_path_col="directory", text_col="transcripts", length_col="duration")


    f.write(nemo_data)
    f.close()

if __name__ == "__main__":
    main()
