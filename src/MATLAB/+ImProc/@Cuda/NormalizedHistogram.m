% NormalizedHistogram - histogram = NormalizedHistogram(imageIn,numBins,min,max,device) 
function histogram = NormalizedHistogram(imageIn,numBins,min,max,device)
    [histogram] = ImProc.Cuda.Mex('NormalizedHistogram',imageIn,numBins,min,max,device);
end
