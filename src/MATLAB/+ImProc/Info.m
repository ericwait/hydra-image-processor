% Info - Get information on all available mex commands.
%    commandInfo = ImProc.);
%    Returns commandInfo structure array containing information on all mex commands.
%       commandInfo.command - Command string
%       commandInfo.outArgs - Cell array of output arguments
%       commandInfo.inArgs - Cell array of input arguments
%       commandInfo.helpLines - Cell array of input arguments
function commandInfo = Info()
     try
          commandInfo = ImProc.Cuda.Info();
     catch errMsg
          commandInfo = ImProc.Local.Info();
          warning(errMsg);
     end
end
