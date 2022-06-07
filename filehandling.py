#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle, json, gzip, re
import pandas as pd
from os import getcwd
from functools import wraps

cwd = getcwd()

def check_gz_extension(filename):
    if re.search("\.gz$", filename):
        zipped_filename = filename
        filename = re.search("(.+)\.gz$", zipped_filename).group(1)
    else:
        zipped_filename = filename + ".gz"
    return (filename, zipped_filename)

def load_from_pkl(filename):
    print("loading data from " + filename, end="...")
    try:
        with open(filename, "rb") as fh:
            data = pickle.load(fh)
        print(" ✅")
    except FileNotFoundError:
        print("\nerror: no file \"" + filename + "\" found in " + cwd)
        return None
    return data

def load_from_zipped_pkl(filename):
    filename, zipped_filename = check_gz_extension(filename)
    print("loading data from " + zipped_filename, end="...")
    try:
        with gzip.open(zipped_filename, "rb") as fh:
            data = pickle.load(fh)
        print(" ✅")
    except FileNotFoundError:
        print("\nerror: no file \"" + zipped_filename + "\" found in " + cwd)
        return None
    return data

def load_from_json(filename):
    print("loading data from " + filename, end="...")
    try:
        with open(filename, "r") as fh:
            data = json.load(fh)
        print(" ✅")
    except FileNotFoundError:
        print("\nerror: no file \"" + filename + "\" found in " + cwd)
        return None
    return data

def load_from_zipped_json(filename):
    filename, zipped_filename = check_gz_extension(filename)
    print("loading data from " + zipped_filename, end="...")
    try:
        with gzip.open(zipped_filename, "rb") as fh:
            data = json.load(fh)
        print(" ✅")
    except FileNotFoundError:
        print("\nerror: no file \"" + zipped_filename + "\" found in " + cwd)
        return None
    return data

def save_to_pkl(data, filename):
    print("saving data to " + filename, end="...")
    with open(filename, "wb") as fh:
        pickle.dump(data, fh)
    print(" ✅")

def save_to_zipped_pkl(data, filename):
    filename, zipped_filename = check_gz_extension(filename)
    print("saving data to " + zipped_filename, end="...")
    with gzip.open(zipped_filename, "wb") as fh:
        pickle.dump(data, fh)
    print(" ✅")

def save_to_json(data, filename):
    print("saving data to " + filename, end="...")
    with open(filename, "w") as fh:
        json.dump(data, fh)
    print(" ✅")

def save_to_zipped_json(data, filename):
    filename, zipped_filename = check_gz_extension(filename)
    print("saving data to " + zipped_filename, end="...")
    with gzip.open(zipped_filename, "wb") as fh:
        fh.write(json.dumps(data).encode("utf-8"))
    print(" ✅")

def load_decorator(load):
    @wraps(load)
    def wrapper(*args, **kwargs):
        print("loading data from " + args[0], end="...")
        try:
            data = load(*args, **kwargs)
            print(" ✅")
        except FileNotFoundError:
            print("\nerror: no file \"" + args[0] + "\" found in " + cwd)
            return None
        return data
    return wrapper

def load_decorator_zip(load):
    @wraps(load)
    def wrapper(*args, **kwargs):
        filename, zipped_filename = check_gz_extension(args[0])
        print("loading data from " + zipped_filename, end="...")
        try:
            data = load(zipped_filename, *args[1:], **kwargs)
            print(" ✅")
        except FileNotFoundError:
            print("\nerror: no file \"" + zipped_filename + "\" found in " + cwd)
            return None
        return data
    return wrapper

def save_decorator(save):
    @wraps(save)
    def wrapper(*args, **kwargs):
        print("saving data to " + args[1], end="...")
        save(*args, **kwargs)
        print(" ✅")
    return wrapper

def save_decorator_zip(save):
    @wraps(save)
    def wrapper(*args, **kwargs):
        filename, zipped_filename = check_gz_extension(args[1])
        print("saving data to " + zipped_filename, end="...")
        save(args[0], zipped_filename, *args[2:], **kwargs)
        print(" ✅")
    return wrapper