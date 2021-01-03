import datetime
import pandas as pd


def create_timestamp():
    """
    Creates timestamp yyyymmdd_hhmm

    :return: string
    """

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    return timestamp


def create_parameters_table(similarity_type, columns, index_column):
    """
    Creates a table with run parameters.

    :param similarity_type: string
    :param columns: string or list of strings
    :param index_column: string
    :return: pandas data frame
    """

    if isinstance(columns, list):
        columns = ', '.join(columns)
    data = [similarity_type, columns, index_column]
    pd_df = pd.DataFrame(columns=['similarity_type', 'columns', 'index_column'], data=[data])
    return pd_df


