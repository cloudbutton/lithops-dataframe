import functools
import re
import inspect


_method_cache = {}


class methodcaller(object):
    """
    Return a callable object that calls the given method on its operand.
    Unlike the builtin `operator.methodcaller`, instances of this class are
    serializable
    """

    __slots__ = ("method",)
    func = property(lambda self: self.method)  # For `funcname` to work

    def __new__(cls, method):
        if method in _method_cache:
            return _method_cache[method]
        self = object.__new__(cls)
        self.method = method
        _method_cache[method] = self
        return self

    def __call__(self, obj, *args, **kwargs):
        return getattr(obj, self.method)(*args, **kwargs)

    def __reduce__(self):
        return (methodcaller, (self.method,))

    def __str__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.method)

    __repr__ = __str__


class MethodCache(object):
    """Attribute access on this object returns a methodcaller for that
    attribute.
    Examples
    --------
    >>> a = [1, 3, 3]
    >>> M.count(a, 3) == a.count(3)
    True
    """

    __getattr__ = staticmethod(methodcaller)
    __dir__ = lambda self: list(_method_cache)


M = MethodCache()


def _skip_doctest(line):
    # NumPy docstring contains cursor and comment only example
    stripped = line.strip()
    if stripped == ">>>" or stripped.startswith(">>> #"):
        return line
    elif ">>>" in stripped and "+SKIP" not in stripped:
        if "# doctest:" in line:
            return line + ", +SKIP"
        else:
            return line + "  # doctest: +SKIP"
    else:
        return line


def skip_doctest(doc):
    if doc is None:
        return ""
    return "\n".join([_skip_doctest(line) for line in doc.split("\n")])


def extra_titles(doc):
    lines = doc.split("\n")
    titles = {
        i: lines[i].strip()
        for i in range(len(lines) - 1)
        if lines[i + 1].strip() and all(c == "-" for c in lines[i + 1].strip())
    }

    seen = set()
    for i, title in sorted(titles.items()):
        if title in seen:
            new_title = "Extra " + title
            lines[i] = lines[i].replace(title, new_title)
            lines[i + 1] = lines[i + 1].replace("-" * len(title), "-" * len(new_title))
        else:
            seen.add(title)

    return "\n".join(lines)


def get_named_args(func):
    """Get all non ``*args/**kwargs`` arguments for a function"""
    s = inspect.signature(func)
    return [
        n
        for n, p in s.parameters.items()
        if p.kind in [p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY, p.KEYWORD_ONLY]
    ]


def ignore_warning(doc, cls, name, extra="", skipblocks=0):
    """Expand docstring by adding disclaimer and extra text"""
    import inspect

    if inspect.isclass(cls):
        l1 = "This docstring was copied from %s.%s.%s.\n\n" % (
            cls.__module__,
            cls.__name__,
            name,
        )
    else:
        l1 = "This docstring was copied from %s.%s.\n\n" % (cls.__name__, name)
    l2 = "Some inconsistencies with the Dask version may exist."

    i = doc.find("\n\n")
    if i != -1:
        # Insert our warning
        head = doc[: i + 2]
        tail = doc[i + 2 :]
        while skipblocks > 0:
            i = tail.find("\n\n")
            head = tail[: i + 2]
            tail = tail[i + 2 :]
            skipblocks -= 1
        # Indentation of next line
        indent = re.match(r"\s*", tail).group(0)
        # Insert the warning, indented, with a blank line before and after
        if extra:
            more = [indent, extra.rstrip("\n") + "\n\n"]
        else:
            more = []
        bits = [head, indent, l1, indent, l2, "\n\n"] + more + [tail]
        doc = "".join(bits)

    return doc


def unsupported_arguments(doc, args):
    """ Mark unsupported arguments with a disclaimer """
    lines = doc.split("\n")
    for arg in args:
        subset = [
            (i, line)
            for i, line in enumerate(lines)
            if re.match(r"^\s*" + arg + " ?:", line)
        ]
        if len(subset) == 1:
            [(i, line)] = subset
            lines[i] = line + "  (Not supported in Dask)"
    return "\n".join(lines)


def _derived_from(cls, method, ua_args=[], extra="", skipblocks=0):
    """ Helper function for derived_from to ease testing """
    # do not use wraps here, as it hides keyword arguments displayed
    # in the doc
    original_method = getattr(cls, method.__name__)

    if isinstance(original_method, property):
        # some things like SeriesGroupBy.unique are generated.
        original_method = original_method.fget

    doc = original_method.__doc__
    if doc is None:
        doc = ""

    # Insert disclaimer that this is a copied docstring
    if doc:
        doc = ignore_warning(
            doc, cls, method.__name__, extra=extra, skipblocks=skipblocks
        )
    elif extra:
        doc += extra.rstrip("\n") + "\n\n"

    # Mark unsupported arguments
    try:
        method_args = get_named_args(method)
        original_args = get_named_args(original_method)
        not_supported = [m for m in original_args if m not in method_args]
    except ValueError:
        not_supported = []
    if len(ua_args) > 0:
        not_supported.extend(ua_args)
    if len(not_supported) > 0:
        doc = unsupported_arguments(doc, not_supported)

    doc = skip_doctest(doc)
    doc = extra_titles(doc)

    return doc


def derived_from(original_klass, version=None, ua_args=[], skipblocks=0):
    """Decorator to attach original class's docstring to the wrapped method.
    The output structure will be: top line of docstring, disclaimer about this
    being auto-derived, any extra text associated with the method being patched,
    the body of the docstring and finally, the list of keywords that exist in
    the original method but not in the dask version.
    Parameters
    ----------
    original_klass: type
        Original class which the method is derived from
    version : str
        Original package version which supports the wrapped method
    ua_args : list
        List of keywords which Dask doesn't support. Keywords existing in
        original but not in Dask will automatically be added.
    skipblocks : int
        How many text blocks (paragraphs) to skip from the start of the
        docstring. Useful for cases where the target has extra front-matter.
    """

    def wrapper(method):
        try:
            extra = getattr(method, "__doc__", None) or ""
            method.__doc__ = _derived_from(
                original_klass,
                method,
                ua_args=ua_args,
                extra=extra,
                skipblocks=skipblocks,
            )
            return method

        except AttributeError:
            module_name = original_klass.__module__.split(".")[0]

            @functools.wraps(method)
            def wrapped(*args, **kwargs):
                msg = "Base package doesn't support '{0}'.".format(method.__name__)
                if version is not None:
                    msg2 = " Use {0} {1} or later to use this method."
                    msg += msg2.format(module_name, version)
                raise NotImplementedError(msg)

            return wrapped

    return wrapper


def parse_bytes(s):
    """Parse byte string to numbers
    >>> from dask.utils import parse_bytes
    >>> parse_bytes('100')
    100
    >>> parse_bytes('100 MB')
    100000000
    >>> parse_bytes('100M')
    100000000
    >>> parse_bytes('5kB')
    5000
    >>> parse_bytes('5.4 kB')
    5400
    >>> parse_bytes('1kiB')
    1024
    >>> parse_bytes('1e6')
    1000000
    >>> parse_bytes('1e6 kB')
    1000000000
    >>> parse_bytes('MB')
    1000000
    >>> parse_bytes(123)
    123
    >>> parse_bytes('5 foos')  # doctest: +SKIP
    ValueError: Could not interpret 'foos' as a byte unit
    """
    if isinstance(s, (int, float)):
        return int(s)
    s = s.replace(" ", "")
    if not any(char.isdigit() for char in s):
        s = "1" + s

    for i in range(len(s) - 1, -1, -1):
        if not s[i].isalpha():
            break
    index = i + 1

    prefix = s[:index]
    suffix = s[index:]

    try:
        n = float(prefix)
    except ValueError as e:
        raise ValueError("Could not interpret '%s' as a number" % prefix) from e

    try:
        multiplier = byte_sizes[suffix.lower()]
    except KeyError as e:
        raise ValueError("Could not interpret '%s' as a byte unit" % suffix) from e

    result = n * multiplier
    return int(result)


byte_sizes = {
    "kB": 10 ** 3,
    "MB": 10 ** 6,
    "GB": 10 ** 9,
    "TB": 10 ** 12,
    "PB": 10 ** 15,
    "KiB": 2 ** 10,
    "MiB": 2 ** 20,
    "GiB": 2 ** 30,
    "TiB": 2 ** 40,
    "PiB": 2 ** 50,
    "B": 1,
    "": 1,
}
byte_sizes = {k.lower(): v for k, v in byte_sizes.items()}
byte_sizes.update({k[0]: v for k, v in byte_sizes.items() if k and "i" not in k})
byte_sizes.update({k[:-1]: v for k, v in byte_sizes.items() if k and "i" in k})


def is_dataframe_like(df):
    """ Looks like a Pandas DataFrame """
    typ = type(df)
    return (
        all(hasattr(typ, name) for name in ("groupby", "head", "merge", "mean"))
        and all(hasattr(df, name) for name in ("dtypes", "columns"))
        and not any(hasattr(typ, name) for name in ("name", "dtype"))
    )


def is_series_like(s):
    """ Looks like a Pandas Series """
    typ = type(s)
    return (
        all(hasattr(typ, name) for name in ("groupby", "head", "mean"))
        and all(hasattr(s, name) for name in ("dtype", "name"))
        and "index" not in typ.__name__.lower()
    )


def is_index_like(s):
    """ Looks like a Pandas Index """
    typ = type(s)
    return (
        all(hasattr(s, name) for name in ("name", "dtype"))
        and "index" in typ.__name__.lower()
    )
