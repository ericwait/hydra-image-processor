% SumArray - sum = SumArray(imageIn,device) 
function sum = SumArray(imageIn,device)
    [sum] = Cuda.Mex('SumArray',imageIn,device);
end
