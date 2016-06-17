% Segment - imageOut = Segment(imageIn,alpha,MorphClosure,device) 
function imageOut = Segment(imageIn,alpha,MorphClosure,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('Segment',imageIn,alpha,MorphClosure,device);

    delete(mutexfile);
end
