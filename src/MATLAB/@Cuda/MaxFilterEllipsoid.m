% MaxFilterEllipsoid - imageOut = MaxFilterEllipsoid(imageIn,radius,device) 
function imageOut = MaxFilterEllipsoid(imageIn,radius,device)
    [imageOut] = Cuda.Mex('MaxFilterEllipsoid',imageIn,radius,device);
end
