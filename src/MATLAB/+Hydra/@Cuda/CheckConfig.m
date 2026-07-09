% CheckConfig - Get Hydra library configuration information.
%    [hydraConfig] = Hydra.Cuda.CheckConfig()
%    Returns hydraConfig structure with configuration information.
%    
function [hydraConfig] = CheckConfig()
    [hydraConfig] = Hydra.Cuda.Hydra('CheckConfig');
end
