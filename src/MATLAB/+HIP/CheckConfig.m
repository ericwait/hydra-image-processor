% CheckConfig - Get Hydra library configuration information.
%    [hydraConfig] = HIP.CheckConfig()
%    Returns hydraConfig structure with configuration information.
%    
function [hydraConfig] = CheckConfig()
    try
        [hydraConfig] = HIP.Cuda.CheckConfig();
    catch errMsg
        warning(errMsg.message);
        [hydraConfig] = HIP.Local.CheckConfig();
    end
end
