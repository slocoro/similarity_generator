import datetime


def create_timestamp():
    """
    Creates timestamp yyyymmdd_hhmm

    :return: string
    """

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    return timestamp

