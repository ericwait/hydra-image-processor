% NormalizedHistogram - histogram = NormalizedHistogram(imageIn,numBins,min,max,device) 
function histogram = NormalizedHistogram(imageIn,numBins,min,max,device)
    [histogram] = Cuda.Mex('NormalizedHistogram',imageIn,numBins,min,max,device);
end
