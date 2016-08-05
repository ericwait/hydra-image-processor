% MedianFilter - imageOut = MedianFilter(imageIn,Neighborhood,device) 
function imageOut = MedianFilter(imageIn,Neighborhood,device)
    [imageOut] = ImProc.Cuda.Mex('MedianFilter',imageIn,Neighborhood,device);
end
