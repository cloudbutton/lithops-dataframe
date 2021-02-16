import pandas as pd
import pandas._libs.lib as lib
import csv

from pandas._typing import StorageOptions

from lithops import Storage
from .core import DataFrame
from lithops.utils import split_object_url
from _io import BytesIO


def read_csv(
    filepath,
    sep=lib.no_default,
    delimiter=None,
    # Column and Index Locations and Names
    header="infer",
    names=None,
    index_col=None,
    usecols=None,
    squeeze=False,
    prefix=None,
    mangle_dupe_cols=True,
    # General Parsing Configuration
    dtype=None,
    engine=None,
    converters=None,
    true_values=None,
    false_values=None,
    skipinitialspace=False,
    skiprows=None,
    skipfooter=0,
    nrows=None,
    # NA and Missing Data Handling
    na_values=None,
    keep_default_na=True,
    na_filter=True,
    verbose=False,
    skip_blank_lines=True,
    # Datetime Handling
    parse_dates=False,
    infer_datetime_format=False,
    keep_date_col=False,
    date_parser=None,
    dayfirst=False,
    cache_dates=True,
    # Iteration
    iterator=False,
    chunksize=None,
    # Quoting, Compression, and File Format
    compression="infer",
    thousands=None,
    decimal: str = ".",
    lineterminator=None,
    quotechar='"',
    quoting=csv.QUOTE_MINIMAL,
    doublequote=True,
    escapechar=None,
    comment=None,
    encoding=None,
    dialect=None,
    # Error Handling
    error_bad_lines=True,
    warn_bad_lines=True,
    # Internal
    delim_whitespace=False,
    memory_map=False,
    float_precision=None,
    storage_options: StorageOptions = None,
    # lithops
    sample=250000,
    npartitions=4
):
    kwds = locals()
    del kwds["filepath"]
    del kwds["sample"]
    del kwds["npartitions"]

    sb, bucket, prefix, obj_name = split_object_url(filepath)
    key = '{}/{}'.format(prefix, obj_name) if prefix else obj_name

    st = Storage(backend=sb)
    data_stream = st.get_object(bucket, key, stream=True)
    data = BytesIO(data_stream.read(sample))

    df = pd.read_csv(data, **kwds)
    return DataFrame(df, filepath, npartitions)
