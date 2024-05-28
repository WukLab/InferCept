import enum
import uuid
from platform import uname

import psutil
import torch
from typing import List

from vllm import cuda_utils


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0
        
class InvalidAccessError(Exception):
    pass

def invalidate_access(field_names):
    def decorator(cls):
        original_getattr = cls.__getattribute__

        def new_getattr(self, name):
            if name in field_names:
                raise InvalidAccessError(f"Access to {name} is invalid")
            return original_getattr(self, name)

        cls.__getattribute__ = new_getattr
        return cls

    return decorator

def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97  # pylint: disable=invalid-name
    max_shared_mem = cuda_utils.get_device_attribute(
        cudaDevAttrMaxSharedMemoryPerBlockOptin, gpu)
    return int(max_shared_mem)


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(uname()).lower()


# TODO: Change this back to API response key when doing the real-case
# NOTE: Currently this stop string is for testing only!
# "not" is the token right after prompt in examples/test_pause.py
def get_api_stop_string() -> str:
    # return 'Integrity'
    # return '\n'
    # return 'a'
    # return "<TOOLFORMER_API_RESPONSE>"
    # return "Editor" # gpt-j
    # return "asa" # baichuan-13b
    # return "mandated" # opt
    return "USE" # dummy llama, vulcuna
    return "not"

def get_api_stop_strings() -> List[str]:
    # return "<TOOLFORMER_API_RESPONSE>"
    return ['\n', 'Editor', 'asa', 'USE'] 

def get_api_stop_token() -> int:
    # react "PAUSE"
    return 17171