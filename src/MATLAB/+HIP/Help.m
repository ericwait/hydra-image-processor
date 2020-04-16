% Help - Print detailed usage information for the specified command.
%    HIP.Help([command])
function Help(command)
    HIP.Cuda.HIP('Help',command);
end
