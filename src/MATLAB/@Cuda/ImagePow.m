% ImagePow - This will raise each voxel value to the power provided.
%    imageOut = Cuda.ImagePow(imageIn,power,device)
%    	ImageIn -- can be an image up to three dimensions and of type (uint8,int8,uint16,int16,uint32,int32,single,double).
%    	Power -- must be a double.
%    	Device -- this is an optional parameter that indicates which Cuda capable device to use.
%    	ImageOut -- will have the same dimensions and type as imageIn. Values are clamped to the range of the image space.
function imageOut = ImagePow(imageIn,power,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('ImagePow',imageIn,power,device);

    delete(mutexfile);
end
