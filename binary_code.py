import numpy as np
import matplotlib.pyplot as plt

ALPHABET = np.asarray(list("abcd"), dtype=object)


def symbols_to_binarray(s, alphabet=ALPHABET, dtype=np.float32):
    """One-hot encode a sequence of symbols
    
    This numerical representation of a string of symbols is useful
    to feed the data and expected labels to the input and output
    layers of recurrent networks.
    """
    alphabet = np.asarray(list(alphabet), dtype=object)
    n_samples = len(s)
    n_features = len(alphabet)

    mapping = dict(zip(alphabet, range(n_features)))
    
    code = np.zeros((n_samples, n_features), dtype=dtype)
    for i, e in enumerate(s):
        code[i, mapping[e]] = 1.0
    return code


def binarray_to_symbols(code, alphabet=ALPHABET):
    """Convert encoded data by to a string of symbols"""
    n_samples, n_features = code.shape
    if n_features != len(alphabet):
        raise ValueError(
            "code should have %d columns (instead of %d)."
            % (len(alphabet), n_features)
        )

    # Make sure that the alphabet is a numpy array of symbols
    # to make it possible to leverage numpy fancy indexing
    if not isinstance(alphabet, np.ndarray):
        alphabet = np.asarray(list(alphabet), dtype='object')

    return "".join(alphabet[code.argmax(axis=1)])


def plot_binary_tape(encoded_sequence, alphabet=ALPHABET):
    plt.matshow(encoded_sequence, cmap=plt.cm.gray)
    plt.xticks(np.arange(len(alphabet)), alphabet)
    

def plot_parallel_tapes(input_data, output_data,
                        input_symbols, output_symbols):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 10))
    for ax, data in zip(axes, [input_data, output_data]):
        ax.matshow(data, cmap=plt.cm.gray)
        ax.set_xticks(np.arange(len(ALPHABET)))
        ax.set_xticklabels(ALPHABET, fontsize=18)
    fig.tight_layout()
    plt.title("input: %r, output: %r" % (input_symbols, output_symbols))