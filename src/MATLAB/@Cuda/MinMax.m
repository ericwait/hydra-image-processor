% MinMax - [min,max] = MinMax(imageIn,device) 
function [min,max] = MinMax(imageIn,device)
    [min,max] = Cuda.Mex('MinMax',imageIn,device);
end
