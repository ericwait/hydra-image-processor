% SumArray - sum = SumArray(imageIn,device) 
function sum = SumArray(imageIn,device)
    [sum] = ImProc.Cuda.Mex('SumArray',imageIn,device);
end
