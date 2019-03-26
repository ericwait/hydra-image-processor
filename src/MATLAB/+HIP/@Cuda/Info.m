% Info - Get information on all available mex commands.
%    [cmdInfo] = HIP.Cuda.Info()
%    Returns commandInfo structure array containing information on all mex commands.
%       commandInfo.command - Command string
%       commandInfo.outArgs - Comma-delimited string list of output arguments
%       commandInfo.inArgs - Comma-delimited string list of input arguments
%       commandInfo.helpLines - Help string
%    
function [cmdInfo] = Info()
    [cmdInfo] = HIP.Cuda.Mex('Info');
end
