% DeviceCount - This will return statistics on the Cuda devices available.
%    [numCudaDevices,memoryStats] = ImProc.Cuda.DeviceCount()
%    	NumCudaDevices -- this is the number of Cuda devices available.
%    	MemoryStats -- this is an array of structures where each entery corresponds to a Cuda device.
%    		The memory structure contains the total memory on the device and the memory available for a Cuda call.
function [numCudaDevices,memoryStats] = DeviceCount()
    [numCudaDevices,memoryStats] = ImProc.Cuda.Mex('DeviceCount');
end
