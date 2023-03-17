import pandas
import pandas as pd


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


def main():

    df = pd.read_csv("torgo_transcripts.csv")

    output = columns_to_txt(df, "directory", "duration")


    f = open("demo_file.txt", "w")
    f.write(output)

    f.close()

if __name__ == "__main__":
    main()
