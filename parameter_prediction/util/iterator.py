import numpy as np

def sliding_window(iterable, n):
    window = []
    for x in iter(iterable):
        window.append(x)
        window = window[-n:]
        if len(window) == n:
            # need to copy the list here or weird things happen
            yield list(window)

def blocks(X, blocks):
    for i in xrange(blocks.max()+1):
        yield X[blocks == i]
