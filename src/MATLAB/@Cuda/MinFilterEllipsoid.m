% MinFilterEllipsoid - imageOut = MinFilterEllipsoid(imageIn,radius,device) 
function imageOut = MinFilterEllipsoid(imageIn,radius,device)
    [imageOut] = Cuda.Mex('MinFilterEllipsoid',imageIn,radius,device);
end
