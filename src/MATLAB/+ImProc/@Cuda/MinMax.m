% MinMax - [min,max] = MinMax(imageIn,device) 
function [min,max] = MinMax(imageIn,device)
    [min,max] = ImProc.Cuda.Mex('MinMax',imageIn,device);
end
