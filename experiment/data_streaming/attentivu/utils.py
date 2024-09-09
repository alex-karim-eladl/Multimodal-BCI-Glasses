import os
import numpy as np
import platform
import subprocess


def write_arr(path, arr, cols=None):
    """Called every so often to dump streamed data to a file

    Parameters
    ----------
    path : str
        Path to file to write.
    arr : numpy.ndarray
        Array of data to write. Shape is (Samples, Columns).
    cols: [str]
        Optional list of names to write as the header for the file.
        If None, no column names are written.
        Length must match the number of columns in arr.

    Returns
    -------
    None
    """
    with open(path, 'a+') as f:
        if os.stat(path).st_size == 0:
            # print("Created file.")
            if cols:
                assert (
                    len(cols) == arr.shape[1]
                ), f"Number of column names doesn't match the number of columns in the array:\n{cols}\n{arr.shape}"
                f.write(','.join(cols) + '\n')
        else:
            np.savetxt(f, arr, fmt='%s', delimiter=',')
        f.close()


def get_macos_version():
    if 'Darwin' != platform.system():
        return (None, None, None)

    # macOS is difficult; SYSTEM_VERSION_COMPAT hides the version number from python
    p = subprocess.Popen('sw_vers', stdout=subprocess.PIPE)
    result = p.communicate()[0].decode('utf-8')
    return tuple(result.split('\n')[1].split('\t')[1].split('.'))


def get_human_version():
    if 'Darwin' != platform.system():
        return platform.release()

    return '.'.join(get_macos_version())
