% MultiplyTwoImages - imageOut = MultiplyTwoImages(imageIn1,imageIn2,factor,device) 
function imageOut = MultiplyTwoImages(imageIn1,imageIn2,factor,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('MultiplyTwoImages',imageIn1,imageIn2,factor,device);

    delete(mutexfile);
end
