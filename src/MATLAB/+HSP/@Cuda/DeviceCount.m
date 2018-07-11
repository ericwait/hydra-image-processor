% DeviceCount - This will return the number of Cuda devices available, and their memory.
%    [numCudaDevices,memoryStats] = HSP.Cuda.DeviceCount()
%    	NumCudaDevices -- this is the number of Cuda devices available.
%    	MemoryStats -- this is an array of structures where each entry corresponds to a Cuda device.
%    		The memory structure contains the total memory on the device and the memory available for a Cuda call.
function [numCudaDevices,memoryStats] = DeviceCount()
    [numCudaDevices,memoryStats] = HSP.Cuda.Mex('DeviceCount');
end
