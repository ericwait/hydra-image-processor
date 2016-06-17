% Help - Help on a specified command.
%    Cuda.Help(command)
%    Print detailed usage information for the specified command.
function Help(command)
    curPath = which('Cuda');
    curPath = fileparts(curPath);
    mutexfile = fullfile(curPath,sprintf('device%02d.txt',device));
    while(exist(mutexfile,'file'))
        pause(1);
    end
    f = fopen(mutexfile,'wt');
    fclose(f);

    Cuda.Mex('Help',command);

    delete(mutexfile);
end
