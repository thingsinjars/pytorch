#!/usr/bin/env python3
import datetime
import json
import signal
import sys
import time
from typing import Any, Dict, List

import psutil  # type: ignore[import]
import pynvml  # type: ignore[import]

# ROCm does not currently have the rocm_smi module installed to a pythonic location.
# Must import from ROCm installation path.
# Cannot use the high-level rocm_smi cmdline module due to its use of exit().
# Must use the lower-level ctypes wrappers exposed through rsmiBindings.
sys.path.append("/opt/rocm/libexec/rocm_smi")
try:
    from ctypes import byref, c_uint32, c_uint64

    from rsmiBindings import (  # type: ignore[import]
        rocmsmi,
        rsmi_process_info_t,
        rsmi_status_t,
    )
except ImportError as e:
    pass


def get_processes_running_python_tests() -> List[Any]:
    """
    iterates through running processes using `psutil` and adds to a list any
    processes that have "python" in their name and execute a command using `cmdline()`.

    Returns:
        List[Any]: a list of Python processes running on the system.

    """
    python_processes = []
    for process in psutil.process_iter():
        try:
            if "python" in process.name() and process.cmdline():
                python_processes.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # access denied or the process died
            pass
    return python_processes


def get_per_process_cpu_info() -> List[Dict[str, Any]]:
    """
    retrieves CPU information and memory usage for each running Python test process,
    and returns a list of dictionaries containing this information for each process.

    Returns:
        List[Dict[str, Any]]: a list of dictionaries containing CPU information
        and memory usage for each running Python process.

    """
    processes = get_processes_running_python_tests()
    per_process_info = []
    for p in processes:
        info = {
            "pid": p.pid,
            "cmd": " ".join(p.cmdline()),
            "cpu_percent": p.cpu_percent(),
            "rss_memory": p.memory_info().rss,
        }

        # https://psutil.readthedocs.io/en/latest/index.html?highlight=memory_full_info
        # requires higher user privileges and could throw AccessDenied error, i.e. mac
        try:
            memory_full_info = p.memory_full_info()

            info["uss_memory"] = memory_full_info.uss
            if "pss" in memory_full_info:
                # only availiable in linux
                info["pss_memory"] = memory_full_info.pss

        except psutil.AccessDenied as e:
            # It's ok to skip this
            pass

        per_process_info.append(info)
    return per_process_info


def get_per_process_gpu_info(handle: Any) -> List[Dict[str, Any]]:
    """
    retrieves GPU information for each running process on a host using the NVML library.

    Args:
        handle (Any): 3D NVML handle that is used to retrieve GPU information for
            each process.

    Returns:
        List[Dict[str, Any]]: a list of dictionaries containing information about
        each running process on the system, including its PID and GPU memory usage.

    """
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    per_process_info = []
    for p in processes:
        info = {"pid": p.pid, "gpu_memory": p.usedGpuMemory}
        per_process_info.append(info)
    return per_process_info


def rocm_ret_ok(ret: int) -> Any:
    return ret == rsmi_status_t.RSMI_STATUS_SUCCESS


def rocm_list_devices() -> List[int]:
    """
    calls the `rocmsmi.rsmi_num_monitor_devices` function to retrieve a list of
    device IDs. It then returns the list of IDs as an integer list.

    Returns:
        List[int]: a list of integer indices representing the available Rocket
        Morty devices.

    """
    num = c_uint32(0)
    ret = rocmsmi.rsmi_num_monitor_devices(byref(num))
    if rocm_ret_ok(ret):
        return list(range(num.value))
    return []


def rocm_get_mem_use(device: int) -> float:
    """
    calculates the memory usage ratio for a given device, returning the result as
    a floating-point number.

    Args:
        device (int): 64-bit integer value of the device for which memory usage
            is to be calculated.

    Returns:
        float: a fraction representing the percentage of memory used by a RoCM device.

    """
    memoryUse = c_uint64()
    memoryTot = c_uint64()

    ret = rocmsmi.rsmi_dev_memory_usage_get(device, 0, byref(memoryUse))
    if rocm_ret_ok(ret):
        ret = rocmsmi.rsmi_dev_memory_total_get(device, 0, byref(memoryTot))
        if rocm_ret_ok(ret):
            return float(memoryUse.value) / float(memoryTot.value)
    return 0.0


def rocm_get_gpu_use(device: int) -> float:
    """
    returns the current percentage of a Rocket Morty GPU's use based on an RMI
    call to `rocmsmi.rsmi_dev_busy_percent_get`.

    Args:
        device (int): 3D accelerator card to be checked for busy percent, and it
            takes an integer value ranging from 0 to 255, inclusive of ROCMSMI
            device IDs.

    Returns:
        float: a percentage value representing the current GPU utilization.

    """
    percent = c_uint32()
    ret = rocmsmi.rsmi_dev_busy_percent_get(device, byref(percent))
    if rocm_ret_ok(ret):
        return float(percent.value)
    return 0.0


def rocm_get_pid_list() -> List[Any]:
    """
    computes and returns a list of process IDs using the Rocks Machine Interface
    (RMI). It takes no arguments and returns a list of process IDs on success, or
    an empty list on failure.

    Returns:
        List[Any]: a list of process IDs.

    """
    num_items = c_uint32()
    ret = rocmsmi.rsmi_compute_process_info_get(None, byref(num_items))
    if rocm_ret_ok(ret):
        buff_sz = num_items.value + 10
        procs = (rsmi_process_info_t * buff_sz)()
        procList = []
        ret = rocmsmi.rsmi_compute_process_info_get(byref(procs), byref(num_items))
        for i in range(num_items.value):
            procList.append(procs[i].process_id)
        return procList
    return []


def rocm_get_per_process_gpu_info() -> List[Dict[str, Any]]:
    """
    retrieves GPU information for each process in a list and returns a list of
    dictionaries containing "pid" and "gpu_memory" fields.

    Returns:
        List[Dict[str, Any]]: a list of dictionaries containing GPU information
        for each process in a given list.

    """
    per_process_info = []
    for pid in rocm_get_pid_list():
        proc = rsmi_process_info_t()
        ret = rocmsmi.rsmi_compute_process_info_by_pid_get(int(pid), byref(proc))
        if rocm_ret_ok(ret):
            info = {"pid": pid, "gpu_memory": proc.vram_usage}
            per_process_info.append(info)
    return per_process_info


if __name__ == "__main__":
    handle = None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError:
        # no pynvml avaliable, probably because not cuda
        pass

    rsmi_handles = []
    try:
        ret = rocmsmi.rsmi_init(0)
        rsmi_handles = rocm_list_devices()
    except Exception:
        # no rocmsmi available, probably because not rocm
        pass

    kill_now = False

    def exit_gracefully(*args: Any) -> None:
        """
        sets a global variable `kill_now` to `True`, indicating that the program
        should exit gracefully with the specified arguments.

        Args:
            	-args (Any): 0 or more arguments passed to the function, which are
                stored in a variable called `kill_now`.

        """
        global kill_now
        kill_now = True

    signal.signal(signal.SIGTERM, exit_gracefully)

    while not kill_now:
        try:
            stats = {
                "time": datetime.datetime.utcnow().isoformat("T") + "Z",
                "total_cpu_percent": psutil.cpu_percent(),
                "per_process_cpu_info": get_per_process_cpu_info(),
            }
            if handle is not None:
                stats["per_process_gpu_info"] = get_per_process_gpu_info(handle)
                # https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html
                gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats["total_gpu_utilization"] = gpu_utilization.gpu
                stats["total_gpu_mem_utilization"] = gpu_utilization.memory
            if rsmi_handles:
                stats["per_process_gpu_info"] = rocm_get_per_process_gpu_info()
                # There are 1 to 4 GPUs in use; these values may sum > 1.0.
                gpu_utilization = 0.0
                gpu_memory = 0.0
                for dev in rsmi_handles:
                    gpu_utilization += rocm_get_gpu_use(dev)
                    gpu_memory += rocm_get_mem_use(dev)
                stats["total_gpu_utilization"] = gpu_utilization
                stats["total_gpu_mem_utilization"] = gpu_memory

        except Exception as e:
            stats = {
                "time": datetime.datetime.utcnow().isoformat("T") + "Z",
                "error": str(e),
            }
        finally:
            print(json.dumps(stats))
            time.sleep(1)
