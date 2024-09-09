from typing import List, Any, Dict, Tuple, Union, Optional, Callable
from pathlib import Path
import json

import hashlib
import ctypes
# import base32_crockford
from filelock import SoftFileLock

from typeguard import typechecked

from data.datasets.dataset import EEGDataset
from preprocessing.preprocessor import Preprocessor


@typechecked
class DatasetManager:
    """
    Filtering, splitting and preprocessing dataset can be quite compute expensive.
    Therefore we want to cache the resulting subsets and variations of the dataset on disk
    so we only need to reload when the same configuration is used again.

    This class is responsible for managing the caching of datasets, checking if a dataset exists, 
    reloading it from disk if it does and applying filtering, preprocessing (and caching) if it doesn't
    """

    def __init__(self,
                 base_data_dir: Union[str, Path],
                 cached_folder_name: str = "cached_datasets") -> None:

        # there will be one json file containing each dataset configuration
        # and the path to the cached dataset, relative to the base_data_dir

        self.base_data_dir = Path(base_data_dir)
        assert self.base_data_dir.exists(), f"Base data directory {self.base_data_dir} does not exist"

        self.cache_file = self.base_data_dir / "cached_datasets.json"
        self.cache_folder = self.base_data_dir / cached_folder_name

        # create cache folder if it doesn't exist
        self.cache_folder.mkdir(parents=True, exist_ok=True)

        # create empty cache file if it doesn't exist
        if not self.cache_file.exists():
            with open(self.cache_file, "w") as f:
                json.dump({}, f)

    @staticmethod
    def locked_call(callable: Callable[[], Any], lock_file: Union[str, Path], lock_timeout: int) -> Any:
        """Locks a callable execution with a given timout and a specified lock file.

        Parameters
        ----------
        callable : Callable[[], Any]
            Whatever should be executed thread safe in a multi host environment.
        lock_file : Union[str, Path]
            The file to lock.
        lock_timeout : int
            The amount of seconds to wait for the lock to be aquired.

        Returns
        -------
        Any
            The result of the callable

        Raises
        ------
        Timeout
            If the locking times out.
        """
        lock_file = Path(lock_file)
        # append .lock to file name
        lock_file = lock_file.parent / (lock_file.name + ".lock")

        lock = SoftFileLock(lock_file, timeout=lock_timeout)
        with lock.acquire(timeout=lock_timeout):
            return callable()

    def load_cached(self, dataset_hash: str) -> Optional[Dict[str, Any]]:
        # read cache file
        with open(self.cache_file, "r") as f:
            cache = json.load(f)
        # check if dataset is cached
        return cache.get(dataset_hash, None)

    @staticmethod
    def hash_dataset_info(dataset_info: Dict[str, Any]) -> str:
        """
        Hashes the dataset info dictionary to a unique string
        """
        assert "base_dataset_path" in dataset_info, "base_dataset_path must be specified in dataset_info"
        return hashlib.md5(json.dumps(dataset_info, sort_keys=True).encode('utf-8')).hexdigest()

    def load_dataset(self,
                     dataset_info: Dict[str, Any],
                     include: Optional[List[str]] = None) -> EEGDataset:
        # check if dataset is cached
        # by default include is none, which means all data is available.
        # if include is specified, only the specified data is available

        dataset_info_w_include = dataset_info.copy()
        dataset_info_w_include["include"] = include

        dataset_hash = DatasetManager.hash_dataset_info(dataset_info)
        dataset_hash_w_include = DatasetManager.hash_dataset_info(dataset_info_w_include)

        # first check if the dataset with the specified include is cached
        cached = DatasetManager.locked_call(
            callable=lambda: self.load_cached(dataset_hash_w_include),
            lock_file=self.cache_file,
            lock_timeout=10
        )
        # if not, check if the dataset without the specified include is cached
        if cached is None:
            print(f"Dataset {dataset_hash_w_include} not cached, checking if {dataset_hash} is cached...")
            cached = DatasetManager.locked_call(
                callable=lambda: self.load_cached(dataset_hash),
                lock_file=self.cache_file,
                lock_timeout=10
            )

            # if not, load it and cache it
            if cached is None:
                print(f"Dataset {dataset_hash} not cached, loading, filtering, preprocessing and caching it now...")
                # load base dataset
                base_dataset_path = self.base_data_dir / dataset_info["base_dataset_path"]
                dataset = EEGDataset.load_from_disk_numpy(base_dataset_path, include=include)

                if dataset_info.get("filters", None) is not None:
                    dataset, remaining_dataset = dataset.split_rulebased(filters=dataset_info["filters"])
                if dataset_info.get("preprocessing", None) is not None:
                    dataset = Preprocessor(dataset, reduced_memory=True).apply(dataset_info["preprocessing"]).to_dataset()
            else:
                print(f"Dataset {dataset_hash} found in cached, loading it now...")
                # load the cached dataset
                dataset_path = Path(self.base_data_dir) / cached["path"]
                dataset = EEGDataset.load_from_disk_numpy(dataset_path, include=include)

            # cache dataset
            DatasetManager.locked_call(
                callable=lambda: self.cache_dataset(dataset_hash_w_include, dataset_info_w_include, dataset),
                lock_file=self.cache_file,
                lock_timeout=10
            )

        cached = DatasetManager.locked_call(
            callable=lambda: self.load_cached(dataset_hash_w_include),
            lock_file=self.cache_file,
            lock_timeout=10
        )
        if cached is None:
            print("asdasd")
        assert cached is not None, "Dataset should be cached now"
        print(f"Dataset {dataset_hash_w_include} found in cached, loading it now...")
        # load the cached dataset
        dataset_path = Path(self.base_data_dir).joinpath(cached["path"])
        return EEGDataset.load_from_disk_numpy(dataset_path, include=include)

    def cache_dataset(self, dataset_hash: str, dataset_info: Dict[str, Any], dataset: EEGDataset) -> None:
        # get cache
        with open(self.cache_file, "r") as f:
            cache = json.load(f)

        assert dataset_hash not in cache, "Dataset hash already in cache, this should not happen"

        # create a unique file name for the dataset
        dataset_id = len(cache) + 1
        dataset_path = self.cache_folder / f"dataset_{dataset_id:03d}"
        # save dataset to disk
        dataset.save_to_disk_numpy(dataset_path)

        cache[dataset_hash] = {
            "path": str(dataset_path.relative_to(self.base_data_dir)),
            "dataset_info": dataset_info
        }

        # write updated cache back to file
        with open(self.cache_file, "w") as f:
            json.dump(cache, f, indent=4)
