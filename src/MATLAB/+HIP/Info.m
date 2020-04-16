% Info - Get information on all available mex commands.
%    [cmdInfo] = HIP.Info()
%    Returns commandInfo structure array containing information on all mex commands.
%       commandInfo.command - Command string
%       commandInfo.outArgs - Comma-delimited string list of output arguments
%       commandInfo.inArgs - Comma-delimited string list of input arguments
%       commandInfo.helpLines - Help string
%    
function [cmdInfo] = Info()
    try
        [cmdInfo] = HIP.Cuda.Info();
    catch errMsg
        warning(errMsg.message);
        [cmdInfo] = HIP.Local.Info();
    end
end
