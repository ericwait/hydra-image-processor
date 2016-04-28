% MarkovRandomFieldDenoiser - imageOut = MarkovRandomFieldDenoiser(imageIn,maxIterations,device) 
function imageOut = MarkovRandomFieldDenoiser(imageIn,maxIterations,device)
    [imageOut] = Cuda.Mex('MarkovRandomFieldDenoiser',imageIn,maxIterations,device);
end
