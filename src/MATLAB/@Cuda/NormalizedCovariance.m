% NormalizedCovariance - normalizedCovariance = NormalizedCovariance(imageIn1,imageIn2,device) 
function normalizedCovariance = NormalizedCovariance(imageIn1,imageIn2,device)
    [normalizedCovariance] = Cuda.Mex('NormalizedCovariance',imageIn1,imageIn2,device);
end
