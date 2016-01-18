function [ImageChunk] =  GetImageChunks(imageDims_rc, numBuffersNeeded, memAvailable, numBitsPerVox, kernelDims_rc)
%[ImageChunk] =  GetImageChunks(imageDims_rc, numBuffersNeeded, memAvailable, numBitsPerVox, kernelDims_rc)
if (~exist('kernelDims_rc','var') || isempty(kernelDims_rc))
    kernelDims_rc = [1,1,1];
end

numVoxels = memAvailable / (numBitsPerVox*numBuffersNeeded);

overlapVolume_rc(1) = (kernelDims_rc(1)-1) * imageDims_rc(2) * imageDims_rc(3);
overlapVolume_rc(2) = imageDims_rc(1) * (kernelDims_rc(2)-1) * imageDims_rc(3);
overlapVolume_rc(3) = imageDims_rc(1) * imageDims_rc(2) * (kernelDims_rc(3)-1);

deviceDims_rc = [0,0,0];

if (overlapVolume_rc(1)>overlapVolume_rc(2) && overlapVolume_rc(1)>overlapVolume_rc(3)) % chunking in Y is the worst
    deviceDims_rc(1) = imageDims_rc(1);
    leftOver = numVoxels/imageDims_rc(1);
    squareDim = floor(sqrt(leftOver));
    
    if (overlapVolume_rc(2)<overlapVolume_rc(3)) % chunking in X is second worst
        if (squareDim>imageDims_rc(2))
            deviceDims_rc(2) = imageDims_rc(2);
        else
            deviceDims_rc(2) = squareDim;
        end
        
        deviceDims_rc(3) = floor(numVoxels/(deviceDims_rc(2)*deviceDims_rc(1)));
        
        if (deviceDims_rc(3)>imageDims_rc(3))
            deviceDims_rc(3) = imageDims_rc(3);
        end
    else % chunking in Z is second worst
        if (squareDim>imageDims_rc(3))
            deviceDims_rc(3) = imageDims_rc(3);
        else
            deviceDims_rc(3) = squareDim;
        end
        
        deviceDims_rc(2) = floor(numVoxels/(deviceDims_rc(3)*deviceDims_rc(1)));
        
        if (deviceDims_rc(2)>imageDims_rc(2))
            deviceDims_rc(2) = imageDims_rc(2);
        end
    end
elseif (overlapVolume_rc(2)>overlapVolume_rc(3)) % chunking in X is the worst
    deviceDims_rc(2) = imageDims_rc(2);
    leftOver = (numVoxels/imageDims_rc(2));
    squareDim = floor(sqrt(leftOver));
    
    if (overlapVolume_rc(1)<overlapVolume_rc(3))
        if (squareDim>imageDims_rc(1))
            deviceDims_rc(1) = imageDims_rc(1);
        else
            deviceDims_rc(1) = squareDim;
        end
        
        deviceDims_rc(3) = floor(numVoxels/(deviceDims_rc(1)*deviceDims_rc(2)));
        
        if (deviceDims_rc(3)>imageDims_rc(3))
            deviceDims_rc(3) = imageDims_rc(3);
        end
    else
        if (squareDim>imageDims_rc(3))
            deviceDims_rc(3) = imageDims_rc(3);
        else
            deviceDims_rc(3) = squareDim;
        end
        
        deviceDims_rc(1) = floor(numVoxels/(deviceDims_rc(3)*deviceDims_rc(2)));
        
        if (deviceDims_rc(1)>imageDims_rc(1))
            deviceDims_rc(1) = imageDims_rc(1);
        end
    end
else % chunking in Z is the worst
    deviceDims_rc(3) = imageDims_rc(3);
    leftOver = (numVoxels/imageDims_rc(3));
    squareDim = floor(sqrt(leftOver));
    
    if (overlapVolume_rc(1)<overlapVolume_rc(2))
        if (squareDim>imageDims_rc(1))
            deviceDims_rc(1) = imageDims_rc(1);
        else
            deviceDims_rc(1) = squareDim;
        end
        
        deviceDims_rc(2) = floor(numVoxels/(deviceDims_rc(1)*deviceDims_rc(3)));
        
        if (deviceDims_rc(2)>imageDims_rc(2))
            deviceDims_rc(2) = imageDims_rc(2);
        end
    else
        if (squareDim>imageDims_rc(2))
            deviceDims_rc(2) = imageDims_rc(2);
        else
            deviceDims_rc(2) = squareDim;
        end
        
        deviceDims_rc(1) = floor(numVoxels/(deviceDims_rc(2)*deviceDims_rc(3)));
        
        if (deviceDims_rc(1)>imageDims_rc(1))
            deviceDims_rc(1) = imageDims_rc(1);
        end
    end
end

ImageChunk = Cuda.Helper.CalcChunks(imageDims_rc, deviceDims_rc, kernelDims_rc);
end
