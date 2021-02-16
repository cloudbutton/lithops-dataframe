import dask as d
import dask.array as da
a = da.arange(10, chunks=2).sum()
#b = da.arange(10, chunks=2).mean()

a.compute()

print(a)
