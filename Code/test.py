from ctypes import WinDLL, CDLL, c_double
from time import time

c_dll = CDLL(r'.\sumArr.dll')

openDevice = c_dll.OpenDevice
closeDevice = c_dll.CloseDevice

gpuFunc = c_dll.GetGpuResult
cpuFunc = c_dll.GetCpuResult

gpuFunc.restype = c_double
cpuFunc.restype = c_double

openDevice()

t0 = time()
ret = gpuFunc()
print(ret, time() - t0)

t0 = time()
ret = cpuFunc()
print(ret, time() - t0)

closeDevice()

print('end')
