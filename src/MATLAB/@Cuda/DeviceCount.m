% DeviceCount - This will return statistics on the Cuda devices available.
%    [numCudaDevices,memoryStats] = Cuda.DeviceCount()
%    	NumCudaDevices -- this is the number of Cuda devices available.
%    	MemoryStats -- this is an array of structures where each entery corresponds to a Cuda device.
%    		The memory structure contains the total memory on the device and the memory available for a Cuda call.
function [numCudaDevices,memoryStats] = DeviceCount()
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [numCudaDevices,memoryStats] = Cuda.Mex('DeviceCount');

    delete(mutexfile);
end
