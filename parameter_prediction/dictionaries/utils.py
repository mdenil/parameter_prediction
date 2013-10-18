import numpy as np
import itertools

def enumerate_space(extent):
    """Enumerate all the points in a space with the given extent.  Use this
    function to ensure spaces are always enumerated in the same order.
    """
    return itertools.imap(np.asarray, itertools.product(*map(xrange, extent)))

