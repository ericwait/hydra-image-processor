% CheckConfig - Get Hydra library configuration information.
%    [hydraConfig] = HIP.Cuda.CheckConfig()
%    Returns hydraConfig structure with configuration information.
%    
function [hydraConfig] = CheckConfig()
    [hydraConfig] = HIP.Cuda.HIP('CheckConfig');
end
