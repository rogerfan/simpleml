'''
Helper functions.
'''
import contextlib

import numpy as np
import numpy.core.arrayprint as arrayprint

__all__ = ()


@contextlib.contextmanager
def np_print_options(strip_zeros=True, **kwargs):
    '''
    Temporarily sets numpy print options (using np.set_printoptions).

    Takes any keyword argument that np.set_printoptions takes and temporarily
    sets them within a 'with' context manager.

    Parameters
    ----------
    strip_zeros : bool, optional
        Set to False to prevent trailing zeros from being stripped
        (default True).
    precision : int, optional
        Number of digits of precision for floating point output (default 8).
    threshold : int, optional
        Total number of array elements which trigger summarization
        rather than full repr (default 1000).
    edgeitems : int, optional
        Number of array items in summary at beginning and end of
        each dimension (default 3).
    linewidth : int, optional
        The number of characters per line for the purpose of inserting
        line breaks (default 75).
    suppress : bool, optional
        Whether or not suppress printing of small floating point values
        using scientific notation (default False).
    nanstr : str, optional
        String representation of floating point not-a-number (default nan).
    infstr : str, optional
        String representation of floating point infinity (default inf).
    formatter : dict of callables, optional
        See np.set_printoptions documentation.
    '''
    origcall = arrayprint.FloatFormat.__call__
    def __call__(self, x, strip_zeros=strip_zeros):
        return origcall.__call__(self, x, strip_zeros)
    arrayprint.FloatFormat.__call__ = __call__
    original = np.get_printoptions()
    np.set_printoptions(**kwargs)
    yield
    np.set_printoptions(**original)
    arrayprint.FloatFormat.__call__ = origcall
