# from: https://raw.githubusercontent.com/andrewgazelka/smart-cache/master/smart_cache/__init__.py
# (MIT License on PyPi)
# has been modified to use super_hash and work on python3.8

import dis
import hashlib
import inspect
import pickle
import queue
import threading
import file_system_py as FS
from os import path
from threading import Thread
from super_hash import super_hash
from super_map import LazyDict


settings = LazyDict(
    default_folder="cache.ignore/",
)

class CacheData:
    calculated = False
    cache_file_name: str
    cache = {}
    deep_hash: str

# since we only care about latest
worker_que = queue.Queue(maxsize=100)

def worker():
    while threading.main_thread().is_alive():
        try:
            data: CacheData = worker_que.get(timeout=0.1) # 0.1 second. Allows for checking if the main thread is alive
            while not worker_que.empty(): # so we only write the latest value
                data = worker_que.get(block=False)
            FS.clear_a_path_for(data.cache_file_name, overwrite=True)
            with open(data.cache_file_name, 'wb') as cache_file:
                pickle.dump((data.deep_hash, data.cache), cache_file, protocol=4)
            worker_que.task_done()
        except queue.Empty:
            continue

thread = Thread(target=worker)
thread.start()

# A thread that consumes data
def consumer(in_q):
    while True:
        # Get some data
        data = in_q.get()
        # Process the data

def cache(folder=settings.default_folder):
    def real_decorator(input_func):
        data = CacheData()  # because we need a reference not a value or compile error
        function_id = super_hash(input_func)
        data.cache_file_name = f'cache.ignore/{function_id}.pickle'
        def wrapper(*args, **kwargs):
            inner_func_args = list(args)
            if not data.calculated:
                data.deep_hash = function_id
                if path.exists(data.cache_file_name):
                    with open(data.cache_file_name, 'rb') as cache_file:
                        func_hash, cache_temp = pickle.load(cache_file)
                        if func_hash == data.deep_hash:
                            data.cache = cache_temp
                data.calculated = True
            arg_hash = super_hash((args, kwargs))
            if arg_hash in data.cache:
                return data.cache[arg_hash]
            result = input_func(*inner_func_args, **kwargs)
            data.cache[arg_hash] = result
            q.put(data, block=False)
            return result
        return wrapper
    return real_decorator