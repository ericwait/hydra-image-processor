% MarkovRandomFieldDenoiser - imageOut = MarkovRandomFieldDenoiser(imageIn,maxIterations,device) 
function imageOut = MarkovRandomFieldDenoiser(imageIn,maxIterations,device)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    [imageOut] = Cuda.Mex('MarkovRandomFieldDenoiser',imageIn,maxIterations,device);

    delete(mutexfile);
end
