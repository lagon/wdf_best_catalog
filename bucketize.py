import collections
import logging as log
import hashlib
import os
import shutil
import typing as t

import tqdm


def bucket_for_filename(filename: str) -> str:
    fname_hash = hashlib.md5(filename.encode()).hexdigest()
    return "".join([fname_hash[0], fname_hash[10]])

def bucketized_filename(filename: str, buckets_root: t.Optional[str]) -> str:
    bucket = bucket_for_filename(filename)
    buckets_root = "" if buckets_root is None else buckets_root
    return os.path.join(buckets_root, bucket, filename)

def ensure_bucket_directory_exists(filename: str, buckets_root: str) -> None:
    bucket = bucket_for_filename(filename)
    directory = os.path.join(buckets_root, bucket)
    os.makedirs(directory, exist_ok=True)
    return

def list_all_buckets(buckets_root: str) -> t.List[str]:
    if not os.path.isdir(buckets_root):
        return []
    buckets: t.List[str] = os.listdir(buckets_root)
    buckets = list(filter(lambda b: os.path.isdir(os.path.join(buckets_root, b)), buckets))
    buckets = list(filter(lambda b: b not in [".", ".."] and not b.startswith("."), buckets))
    return buckets

def list_files_in_buckets(buckets_root: str) -> t.List[str]:
    buckets = list_all_buckets(buckets_root)
    all_files = []
    for buck in tqdm.tqdm(buckets, ncols=100, desc="Listing buckets"):
        files = os.listdir(os.path.join(buckets_root, buck))
        files = list(filter(lambda f: not f.startswith("."), files))
        files = [os.path.join(buckets_root, buck, f) for f in files]
        # files = list(filter(lambda f: os.path.isfile(f), files))
        all_files.extend(files)

    return all_files


def bucketize_directory(directory: str, buckets_root: str) -> t.List[t.Tuple[str, str]]:
    filenames = os.listdir(directory)
    filenames = list(filter(lambda f: not f.startswith("."), filenames))
    all_files_and_buckets = []
    for fn in tqdm.tqdm(filenames, ncols=100, desc="Finding locations for all files"):
        orig_fn = os.path.join(directory, fn)
        if os.path.isfile(orig_fn):
            buck_fn = bucketized_filename(fn, buckets_root)
            all_files_and_buckets.append((orig_fn, buck_fn))
    return all_files_and_buckets

def copy_files_into_buckets(all_files_and_buckets: t.List[t.Tuple[str, str]]) -> None:
    for orig_file, bucketized_file in tqdm.tqdm(all_files_and_buckets, ncols=100, desc="Copying files to buckets"):
        bucket_path = os.path.dirname(bucketized_file)
        os.makedirs(bucket_path, exist_ok=True)
        shutil.copy(orig_file, bucketized_file)

def get_distribution(directory: str) -> collections.Counter:
    dir_counter: collections.Counter = collections.Counter()
    dir_files = os.listdir(directory)
    for filename in tqdm.tqdm(dir_files):
        dir_counter.update(collections.Counter([bucket_for_filename(filename)]))

    return dir_counter
