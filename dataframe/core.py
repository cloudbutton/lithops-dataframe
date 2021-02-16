import pandas as pd
import io
import lithops
from .utils import derived_from, is_series_like, M

no_default = "__no_default__"


class DataFrame:
    def __init__(self, df, filepath, npartitions):
        self.filepath = filepath
        self.df = df
        self.npartitions = npartitions

    def reduction(
        self,
        chunk,
        aggregate=None,
        combine=None,
        meta=no_default,
        token=None,
        split_every=None,
        chunk_kwargs=None,
        aggregate_kwargs=None,
        combine_kwargs=None,
        **kwargs,
    ):
        """Generic row-wise reductions.
        Parameters
        ----------
        chunk : callable
            Function to operate on each partition. Should return a
            ``pandas.DataFrame``, ``pandas.Series``, or a scalar.
        aggregate : callable, optional
            Function to operate on the concatenated result of ``chunk``. If not
            specified, defaults to ``chunk``. Used to do the final aggregation
            in a tree reduction.
            The input to ``aggregate`` depends on the output of ``chunk``.
            If the output of ``chunk`` is a:
            - scalar: Input is a Series, with one row per partition.
            - Series: Input is a DataFrame, with one row per partition. Columns
              are the rows in the output series.
            - DataFrame: Input is a DataFrame, with one row per partition.
              Columns are the columns in the output dataframes.
            Should return a ``pandas.DataFrame``, ``pandas.Series``, or a
            scalar.
        combine : callable, optional
            Function to operate on intermediate concatenated results of
            ``chunk`` in a tree-reduction. If not provided, defaults to
            ``aggregate``. The input/output requirements should match that of
            ``aggregate`` described above.
        $META
        token : str, optional
            The name to use for the output keys.
        split_every : int, optional
            Group partitions into groups of this size while performing a
            tree-reduction. If set to False, no tree-reduction will be used,
            and all intermediates will be concatenated and passed to
            ``aggregate``. Default is 8.
        chunk_kwargs : dict, optional
            Keyword arguments to pass on to ``chunk`` only.
        aggregate_kwargs : dict, optional
            Keyword arguments to pass on to ``aggregate`` only.
        combine_kwargs : dict, optional
            Keyword arguments to pass on to ``combine`` only.
        kwargs :
            All remaining keywords will be passed to ``chunk``, ``combine``,
            and ``aggregate``.
        Examples
        --------
        >>> import pandas as pd
        >>> import dask.dataframe as dd
        >>> df = pd.DataFrame({'x': range(50), 'y': range(50, 100)})
        >>> ddf = dd.from_pandas(df, npartitions=4)
        Count the number of rows in a DataFrame. To do this, count the number
        of rows in each partition, then sum the results:
        >>> res = ddf.reduction(lambda x: x.count(),
        ...                     aggregate=lambda x: x.sum())
        >>> res.compute()
        x    50
        y    50
        dtype: int64
        Count the number of rows in a Series with elements greater than or
        equal to a value (provided via a keyword).
        >>> def count_greater(x, value=0):
        ...     return (x >= value).sum()
        >>> res = ddf.x.reduction(count_greater, aggregate=lambda x: x.sum(),
        ...                       chunk_kwargs={'value': 25})
        >>> res.compute()
        25
        Aggregate both the sum and count of a Series at the same time:
        >>> def sum_and_count(x):
        ...     return pd.Series({'count': x.count(), 'sum': x.sum()},
        ...                      index=['count', 'sum'])
        >>> res = ddf.x.reduction(sum_and_count, aggregate=lambda x: x.sum())
        >>> res.compute()
        count      50
        sum      1225
        dtype: int64
        Doing the same, but for a DataFrame. Here ``chunk`` returns a
        DataFrame, meaning the input to ``aggregate`` is a DataFrame with an
        index with non-unique entries for both 'x' and 'y'. We groupby the
        index, and sum each group to get the final result.
        >>> def sum_and_count(x):
        ...     return pd.DataFrame({'count': x.count(), 'sum': x.sum()},
        ...                         columns=['count', 'sum'])
        >>> res = ddf.reduction(sum_and_count,
        ...                     aggregate=lambda x: x.groupby(level=0).sum())
        >>> res.compute()
           count   sum
        x     50  1225
        y     50  3725
        """
        if aggregate is None:
            aggregate = chunk

        if combine is None:
            if combine_kwargs:
                raise ValueError("`combine_kwargs` provided with no `combine`")
            combine = aggregate
            combine_kwargs = aggregate_kwargs

        chunk_kwargs = chunk_kwargs.copy() if chunk_kwargs else {}
        chunk_kwargs["aca_chunk"] = chunk

        combine_kwargs = combine_kwargs.copy() if combine_kwargs else {}
        combine_kwargs["aca_combine"] = combine

        aggregate_kwargs = aggregate_kwargs.copy() if aggregate_kwargs else {}
        aggregate_kwargs["aca_aggregate"] = aggregate

        return aca(
            self,
            chunk=_reduction_chunk,
            aggregate=_reduction_aggregate,
            combine=_reduction_combine,
            meta=meta,
            token=token,
            split_every=split_every,
            chunk_kwargs=chunk_kwargs,
            aggregate_kwargs=aggregate_kwargs,
            combine_kwargs=combine_kwargs,
            **kwargs,
        )

    def _reduction_agg(self, name, axis=None, skipna=True, split_every=False, out=None):
        axis = self._validate_axis(axis)

        meta = getattr(self._meta_nonempty, name)(axis=axis, skipna=skipna)
        token = self._token_prefix + name

        method = getattr(M, name)
        if axis == 1:
            result = self.map_partitions(
                method, meta=meta, token=token, skipna=skipna, axis=axis
            )
            return handle_out(out, result)
        else:
            result = self.reduction(
                method,
                meta=meta,
                token=token,
                skipna=skipna,
                axis=axis,
                split_every=split_every,
            )
            if isinstance(self, DataFrame):
                result.divisions = (self.columns.min(), self.columns.max())
            return handle_out(out, result)

    def apply(
        self,
        func,
        axis=0,
        broadcast=None,
        raw=False,
        reduce=None,
        args=(),
        meta=no_default,
        result_type=None,
        **kwds,
    ):
        """Parallel version of pandas.DataFrame.apply

        This mimics the pandas version except for the following:

        1.  Only ``axis=1`` is supported (and must be specified explicitly).
        2.  The user should provide output metadata via the `meta` keyword.

        Parameters
        ----------
        func : function
            Function to apply to each column/row
        axis : {0 or 'index', 1 or 'columns'}, default 0
            - 0 or 'index': apply function to each column (NOT SUPPORTED)
            - 1 or 'columns': apply function to each row
        $META
        args : tuple
            Positional arguments to pass to function in addition to the array/series

        Additional keyword arguments will be passed as keywords to the function

        Returns
        -------
        applied : Series or DataFrame

        Examples
        --------
        >>> import pandas as pd
        >>> import dask.dataframe as dd
        >>> df = pd.DataFrame({'x': [1, 2, 3, 4, 5],
        ...                    'y': [1., 2., 3., 4., 5.]})
        >>> ddf = dd.from_pandas(df, npartitions=2)

        Apply a function to row-wise passing in extra arguments in ``args`` and
        ``kwargs``:

        >>> def myadd(row, a, b=1):
        ...     return row.sum() + a + b
        >>> res = ddf.apply(myadd, axis=1, args=(2,), b=1.5)  # doctest: +SKIP

        By default, dask tries to infer the output metadata by running your
        provided function on some fake data. This works well in many cases, but
        can sometimes be expensive, or even fail. To avoid this, you can
        manually specify the output metadata with the ``meta`` keyword. This
        can be specified in many forms, for more information see
        ``dask.dataframe.utils.make_meta``.

        Here we specify the output is a Series with name ``'x'``, and dtype
        ``float64``:

        >>> res = ddf.apply(myadd, axis=1, args=(2,), b=1.5, meta=('x', 'f8'))

        In the case where the metadata doesn't change, you can also pass in
        the object itself directly:

        >>> res = ddf.apply(lambda row: row + 1, axis=1, meta=ddf)

        See Also
        --------
        dask.DataFrame.map_partitions
        """

        pandas_kwargs = {"axis": axis, "raw": raw, "result_type": result_type}

        if axis == 0:
            msg = (
                "lithops.DataFrame.apply only supports axis=1\n"
                "  Try: df.apply(func, axis=1)"
            )
            raise NotImplementedError(msg)

        def pandas_apply_function(obj):
            buf = io.BytesIO(obj.data_stream.read())
            df = pd.read_csv(buf)
            df.apply(func, args=args, **kwds, **pandas_kwargs)

        fexec = lithops.FunctionExecutor()
        fexec.map(pandas_apply_function, [self.filepath], chunk_n=self.npartitions)
        fexec.wait()

        return self

    @derived_from(pd.DataFrame)
    def all(self, axis=None, skipna=True, split_every=False, out=None):
        def pandas_all_function(obj):
            buf = io.BytesIO(obj.data_stream.read())
            df = pd.read_csv(buf)
            df.all(axis=axis, skipna=skipna)

        fexec = lithops.FunctionExecutor()
        fexec.map(pandas_all_function, [self.filepath], chunk_n=self.npartitions)
        fexec.wait()

        return self

    @derived_from(pd.DataFrame)
    def any(self, axis=None, skipna=True, split_every=False, out=None):
        def pandas_any_function(obj):
            buf = io.BytesIO(obj.data_stream.read())
            df = pd.read_csv(buf)
            df.any(axis=axis, skipna=skipna)

        fexec = lithops.FunctionExecutor()
        fexec.map(pandas_any_function, [self.filepath], chunk_n=self.npartitions)
        fexec.wait()

        return self

    @derived_from(pd.DataFrame)
    def sum(
        self,
        axis=None,
        skipna=True,
        split_every=False,
        dtype=None,
        out=None,
        min_count=None,
    ):
        # use self._reduction_agg()
        def pandas_sum_function(obj):
            buf = io.BytesIO(obj.data_stream.read())
            df = pd.read_csv(buf)
            df.sum(axis=axis, skipna=skipna, min_count=min_count)

        fexec = lithops.FunctionExecutor()
        fexec.map(pandas_sum_function, [self.filepath], chunk_n=self.npartitions)
        fexec.wait()

        return self

    @derived_from(pd.DataFrame)
    def prod(
        self,
        axis=None,
        skipna=True,
        split_every=False,
        dtype=None,
        out=None,
        min_count=None,
    ):
        # use self._reduction_agg()
        def pandas_prod_function(obj):
            buf = io.BytesIO(obj.data_stream.read())
            df = pd.read_csv(buf)
            df.prod(axis=axis, skipna=skipna)

        fexec = lithops.FunctionExecutor()
        fexec.map(pandas_prod_function, [self.filepath], chunk_n=self.npartitions)
        fexec.wait()

        return self

    @derived_from(pd.DataFrame)
    def max(self, axis=None, skipna=True, split_every=False, out=None):
        # use self._reduction_agg()
        def pandas_max_function(obj):
            buf = io.BytesIO(obj.data_stream.read())
            df = pd.read_csv(buf)
            df.max(axis=axis, skipna=skipna)

        fexec = lithops.FunctionExecutor()
        fexec.map(pandas_max_function, [self.filepath], chunk_n=self.npartitions)
        fexec.wait()

        return self

    @derived_from(pd.DataFrame)
    def min(self, axis=None, skipna=True, split_every=False, out=None):
        # use self._reduction_agg()
        def pandas_min_function(obj):
            buf = io.BytesIO(obj.data_stream.read())
            df = pd.read_csv(buf)
            df.min(axis=axis, skipna=skipna)

        fexec = lithops.FunctionExecutor()
        fexec.map(pandas_min_function, [self.filepath], chunk_n=self.npartitions)
        fexec.wait()

        return self

    @derived_from(pd.DataFrame)
    def count(self, axis=None, split_every=False):
        # use self.map_partition whens axis = 1 , self.reduction when axis = 0()
        def pandas_count_function(obj):
            buf = io.BytesIO(obj.data_stream.read())
            df = pd.read_csv(buf)
            df.count(axis=axis)

        fexec = lithops.FunctionExecutor()
        fexec.map(pandas_count_function, [self.filepath], chunk_n=self.npartitions)
        fexec.wait()

        return self

    @derived_from(pd.DataFrame)
    def mean(self, axis=None, skipna=True, split_every=False, dtype=None, out=None):
        # use self.map_partition whens axis = 1 , self.reduction when axis = 0()
        def pandas_mean_function(obj):
            buf = io.BytesIO(obj.data_stream.read())
            df = pd.read_csv(buf)
            df.mean(axis=axis, skipna=skipna)

        fexec = lithops.FunctionExecutor()
        fexec.map(pandas_count_function, [self.filepath], chunk_n=self.npartitions)
        fexec.wait()

        return self

    @derived_from(pd.DataFrame)
    def std(
        self, axis=None, skipna=True, ddof=1, split_every=False, dtype=None, out=None
    ):
        # use self.map_partition
        def pandas_count_function(obj):
            buf = io.BytesIO(obj.data_stream.read())
            df = pd.read_csv(buf)
            df.count(axis=axis)

        fexec = lithops.FunctionExecutor()
        fexec.map(pandas_count_function, [self.filepath], chunk_n=self.npartitions)
        fexec.wait()

        return self


def map_partitions(
    func,
    *args,
    meta=no_default,
    enforce_metadata=True,
    transform_divisions=True,
    **kwargs,
):
    """Apply Python function on each DataFrame partition.
    Parameters
    ----------
    func : function
        Function applied to each partition.
    args, kwargs :
        Arguments and keywords to pass to the function.  At least one of the
        args should be a Dask.dataframe. Arguments and keywords may contain
        ``Scalar``, ``Delayed`` or regular python objects. DataFrame-like args
        (both dask and pandas) will be repartitioned to align (if necessary)
        before applying the function.
    enforce_metadata : bool
        Whether or not to enforce the structure of the metadata at runtime.
        This will rename and reorder columns for each partition,
        and will raise an error if this doesn't work or types don't match.
    $META
    """
    name = kwargs.pop("token", None)

    if has_keyword(func, "partition_info"):
        kwargs["partition_info"] = {"number": -1, "divisions": None}

    assert callable(func)
    if name is not None:
        token = tokenize(meta, *args, **kwargs)
    else:
        name = funcname(func)
        token = tokenize(func, meta, *args, **kwargs)
    name = "{0}-{1}".format(name, token)

    from .multi import _maybe_align_partitions

    args = _maybe_from_pandas(args)
    args = _maybe_align_partitions(args)
    dfs = [df for df in args if isinstance(df, _Frame)]
    meta_index = getattr(make_meta(dfs[0]), "index", None) if dfs else None

    if meta is no_default:
        # Use non-normalized kwargs here, as we want the real values (not
        # delayed values)
        meta = _emulate(func, *args, udf=True, **kwargs)
    else:
        meta = make_meta(meta, index=meta_index)

    if has_keyword(func, "partition_info"):
        kwargs["partition_info"] = "__dummy__"

    if all(isinstance(arg, Scalar) for arg in args):
        layer = {
            (name, 0): (apply, func, (tuple, [(arg._name, 0) for arg in args]), kwargs)
        }
        graph = HighLevelGraph.from_collections(name, layer, dependencies=args)
        return Scalar(graph, name, meta)
    elif not (has_parallel_type(meta) or is_arraylike(meta) and meta.shape):
        # If `meta` is not a pandas object, the concatenated results will be a
        # different type
        meta = make_meta(_concat([meta]), index=meta_index)

    # Ensure meta is empty series
    meta = make_meta(meta)

    args2 = []
    dependencies = []
    for arg in args:
        if isinstance(arg, _Frame):
            args2.append(arg)
            dependencies.append(arg)
            continue
        arg = normalize_arg(arg)
        arg2, collections = unpack_collections(arg)
        if collections:
            args2.append(arg2)
            dependencies.extend(collections)
        else:
            args2.append(arg)

    kwargs3 = {}
    simple = True
    for k, v in kwargs.items():
        v = normalize_arg(v)
        v, collections = unpack_collections(v)
        dependencies.extend(collections)
        kwargs3[k] = v
        if collections:
            simple = False

    if enforce_metadata:
        dsk = partitionwise_graph(
            apply_and_enforce,
            name,
            *args2,
            dependencies=dependencies,
            _func=func,
            _meta=meta,
            **kwargs3,
        )
    else:
        kwargs4 = kwargs if simple else kwargs3
        dsk = partitionwise_graph(
            func, name, *args2, **kwargs4, dependencies=dependencies
        )

    divisions = dfs[0].divisions
    if transform_divisions and isinstance(dfs[0], Index) and len(dfs) == 1:
        try:
            divisions = func(
                *[pd.Index(a.divisions) if a is dfs[0] else a for a in args], **kwargs
            )
            if isinstance(divisions, pd.Index):
                divisions = methods.tolist(divisions)
        except Exception:
            pass
        else:
            if not valid_divisions(divisions):
                divisions = [None] * (dfs[0].npartitions + 1)

    if has_keyword(func, "partition_info"):
        dsk = dict(dsk)

        for k, v in dsk.items():
            vv = v
            v = v[0]
            number = k[-1]
            assert isinstance(number, int)
            info = {"number": number, "division": divisions[number]}
            v = copy.copy(v)  # Need to copy and unpack subgraph callable
            v.dsk = copy.copy(v.dsk)
            [(key, task)] = v.dsk.items()
            task = subs(task, {"__dummy__": info})
            v.dsk[key] = task
            dsk[k] = (v,) + vv[1:]

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=dependencies)
    return new_dd_object(graph, name, meta, divisions)


def apply_concat_apply(
    args,
    chunk=None,
    aggregate=None,
    combine=None,
    meta=no_default,
    token=None,
    chunk_kwargs=None,
    aggregate_kwargs=None,
    combine_kwargs=None,
    split_every=None,
    split_out=None,
    split_out_setup=None,
    split_out_setup_kwargs=None,
    sort=None,
    ignore_index=False,
    **kwargs,
):
    """Apply a function to blocks, then concat, then apply again
    Parameters
    ----------
    args :
        Positional arguments for the `chunk` function. All `dask.dataframe`
        objects should be partitioned and indexed equivalently.
    chunk : function [block-per-arg] -> block
        Function to operate on each block of data
    aggregate : function concatenated-block -> block
        Function to operate on the concatenated result of chunk
    combine : function concatenated-block -> block, optional
        Function to operate on intermediate concatenated results of chunk
        in a tree-reduction. If not provided, defaults to aggregate.
    $META
    token : str, optional
        The name to use for the output keys.
    chunk_kwargs : dict, optional
        Keywords for the chunk function only.
    aggregate_kwargs : dict, optional
        Keywords for the aggregate function only.
    combine_kwargs : dict, optional
        Keywords for the combine function only.
    split_every : int, optional
        Group partitions into groups of this size while performing a
        tree-reduction. If set to False, no tree-reduction will be used,
        and all intermediates will be concatenated and passed to ``aggregate``.
        Default is 8.
    split_out : int, optional
        Number of output partitions. Split occurs after first chunk reduction.
    split_out_setup : callable, optional
        If provided, this function is called on each chunk before performing
        the hash-split. It should return a pandas object, where each row
        (excluding the index) is hashed. If not provided, the chunk is hashed
        as is.
    split_out_setup_kwargs : dict, optional
        Keywords for the `split_out_setup` function only.
    sort : bool, default None
        If allowed, sort the keys of the output aggregation.
    ignore_index : bool, default False
        If True, do not preserve index values throughout ACA operations.
    kwargs :
        All remaining keywords will be passed to ``chunk``, ``aggregate``, and
        ``combine``.
    Examples
    --------
    >>> def chunk(a_block, b_block):
    ...     pass
    >>> def agg(df):
    ...     pass
    >>> apply_concat_apply([a, b], chunk=chunk, aggregate=agg)  # doctest: +SKIP
    """
    if chunk_kwargs is None:
        chunk_kwargs = dict()
    if aggregate_kwargs is None:
        aggregate_kwargs = dict()
    chunk_kwargs.update(kwargs)
    aggregate_kwargs.update(kwargs)

    if combine is None:
        if combine_kwargs:
            raise ValueError("`combine_kwargs` provided with no `combine`")
        combine = aggregate
        combine_kwargs = aggregate_kwargs
    else:
        if combine_kwargs is None:
            combine_kwargs = dict()
        combine_kwargs.update(kwargs)

    if not isinstance(args, (tuple, list)):
        args = [args]

    dfs = [arg for arg in args if isinstance(arg, _Frame)]

    npartitions = set(arg.npartitions for arg in dfs)
    if len(npartitions) > 1:
        raise ValueError("All arguments must have same number of partitions")
    npartitions = npartitions.pop()

    if split_every is None:
        split_every = 8
    elif split_every is False:
        split_every = npartitions
    elif split_every < 2 or not isinstance(split_every, Integral):
        raise ValueError("split_every must be an integer >= 2")

    token_key = tokenize(
        token or (chunk, aggregate),
        meta,
        args,
        chunk_kwargs,
        aggregate_kwargs,
        combine_kwargs,
        split_every,
        split_out,
        split_out_setup,
        split_out_setup_kwargs,
    )

    # Chunk
    a = "{0}-chunk-{1}".format(token or funcname(chunk), token_key)
    if len(args) == 1 and isinstance(args[0], _Frame) and not chunk_kwargs:
        dsk = {
            (a, 0, i, 0): (chunk, key) for i, key in enumerate(args[0].__dask_keys__())
        }
    else:
        dsk = {
            (a, 0, i, 0): (
                apply,
                chunk,
                [(x._name, i) if isinstance(x, _Frame) else x for x in args],
                chunk_kwargs,
            )
            for i in range(npartitions)
        }

    # Split
    if split_out and split_out > 1:
        split_prefix = "split-%s" % token_key
        shard_prefix = "shard-%s" % token_key
        for i in range(npartitions):
            dsk[(split_prefix, i)] = (
                hash_shard,
                (a, 0, i, 0),
                split_out,
                split_out_setup,
                split_out_setup_kwargs,
                ignore_index,
            )
            for j in range(split_out):
                dsk[(shard_prefix, 0, i, j)] = (getitem, (split_prefix, i), j)
        a = shard_prefix
    else:
        split_out = 1

    # Combine
    b = "{0}-combine-{1}".format(token or funcname(combine), token_key)
    k = npartitions
    depth = 0
    while k > split_every:
        for part_i, inds in enumerate(partition_all(split_every, range(k))):
            for j in range(split_out):
                conc = (_concat, [(a, depth, i, j) for i in inds], ignore_index)
                if combine_kwargs:
                    dsk[(b, depth + 1, part_i, j)] = (
                        apply,
                        combine,
                        [conc],
                        combine_kwargs,
                    )
                else:
                    dsk[(b, depth + 1, part_i, j)] = (combine, conc)
        k = part_i + 1
        a = b
        depth += 1

    if sort is not None:
        if sort and split_out > 1:
            raise NotImplementedError(
                "Cannot guarantee sorted keys for `split_out>1`."
                " Try using split_out=1, or grouping with sort=False."
            )
        aggregate_kwargs = aggregate_kwargs or {}
        aggregate_kwargs["sort"] = sort

    # Aggregate
    for j in range(split_out):
        b = "{0}-agg-{1}".format(token or funcname(aggregate), token_key)
        conc = (_concat, [(a, depth, i, j) for i in range(k)], ignore_index)
        if aggregate_kwargs:
            dsk[(b, j)] = (apply, aggregate, [conc], aggregate_kwargs)
        else:
            dsk[(b, j)] = (aggregate, conc)

    if meta is no_default:
        meta_chunk = _emulate(chunk, *args, udf=True, **chunk_kwargs)
        meta = _emulate(
            aggregate, _concat([meta_chunk], ignore_index), udf=True, **aggregate_kwargs
        )
    meta = make_meta(
        meta, index=(getattr(make_meta(dfs[0]), "index", None) if dfs else None)
    )

    graph = HighLevelGraph.from_collections(b, dsk, dependencies=dfs)

    divisions = [None] * (split_out + 1)

    return new_dd_object(graph, b, meta, divisions)


aca = apply_concat_apply


def _reduction_chunk(x, aca_chunk=None, **kwargs):
    o = aca_chunk(x, **kwargs)
    # Return a dataframe so that the concatenated version is also a dataframe
    return o.to_frame().T if is_series_like(o) else o


def _reduction_combine(x, aca_combine=None, **kwargs):
    if isinstance(x, list):
        x = pd.Series(x)
    o = aca_combine(x, **kwargs)
    # Return a dataframe so that the concatenated version is also a dataframe
    return o.to_frame().T if is_series_like(o) else o


def _reduction_aggregate(x, aca_aggregate=None, **kwargs):
    if isinstance(x, list):
        x = pd.Series(x)
    return aca_aggregate(x, **kwargs)
