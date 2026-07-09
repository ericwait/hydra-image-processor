% DeviceStats - This will return the statistics of each Cuda capable device installed.
%    [deviceStatsArray] = Hydra.Cuda.DeviceStats()
%    	DeviceStatsArray -- this is an array of structs, one struct per device.
%    		The struct has these fields: name, major, minor, constMem, sharedMem, totalMem, tccDriver, mpCount, threadsPerMP, warpSize, maxThreads.
function [deviceStatsArray] = DeviceStats()
    [deviceStatsArray] = Hydra.Cuda.Hydra('DeviceStats');
end
