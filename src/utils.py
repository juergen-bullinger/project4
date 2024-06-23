#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for pickling / unpickling objects
Created on Thu Apr 18 14:53:32 2024

@author: juergen
"""

from pathlib import Path
import pickle

def pickle_dump_object(pickle_obj, file_name_str):
    """
    Helper function to pickle an object to a file given as a string

    Parameters
    ----------
    pickle_obj : TYPE
        Object to be pickeld.
    file_name_str : str or Path
        Name of the file to which the file should be written.

    Returns
    -------
    None.
    """
    print(f"pickeling to {file_name_str}")
    with Path(file_name_str).open("wb") as fp_pickle:
        pickle.dump(pickle_obj, fp_pickle)


def pickle_load_object(file_name_str):
    """
    Helper function to load a pickled object from a file given as a string

    Parameters
    ----------
    file_name_str : str or Path
        Name of the file to which the file should be written.

    Returns
    -------
    pickle_object.
    """
    with Path(file_name_str).open("rb") as fp_pickle:
        pickle_obj = pickle.load(fp_pickle)
    return pickle_obj

