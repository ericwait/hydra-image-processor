function packageToolboxCI(version)
% packageToolboxCI Package the Hydra Image Processor toolbox (CI entry point).
%
%   packageToolboxCI(VERSION) builds "Hydra Image Processor.mltbx" in
%   src/MATLAB from the +HIP package and CudaMexTester.m. The MEX binaries
%   (HIP.mexw64 / HIP.mexa64) must already be present in +HIP/@Cuda.
%
%   VERSION is single-sourced from pyproject.toml by the workflow; the GUID
%   below matches HydraImageProcessor.prj so installs upgrade in place. The
%   .prj itself is not used here because it embeds machine-specific absolute
%   paths - it remains the entry point for interactive packaging in the IDE.

    if nargin < 1 || isempty(version)
        error('packageToolboxCI:noVersion', 'A toolbox version string is required.');
    end

    scriptDir = fileparts(mfilename('fullpath'));
    matlabDir = fileparts(scriptDir);                 % src/MATLAB
    repoRoot  = fileparts(fileparts(matlabDir));      % repository root

    guid = '8c91e1c9-d55d-4c1e-96df-875fd68e82cd';    % keep in sync with HydraImageProcessor.prj
    opts = matlab.addons.toolbox.ToolboxOptions(matlabDir, guid);

    opts.ToolboxName    = 'Hydra Image Processor';
    opts.ToolboxVersion = version;
    opts.AuthorName     = 'Eric Wait';
    opts.AuthorEmail    = 'eric@waitphoto.com';
    opts.Summary        = ['Hydra Image Processor handles large 1D to 5D image data by ' ...
                           'optimally chunking across GPUs, ensuring energy-insulated boundaries.'];
    opts.Description    = sprintf(['Hydra Image Processor (Hydra) is a collection of signal filters ' ...
                           'for image analysis that can handle 1-5 dimensional data (x, y, z, ' ...
                           'channels, time), efficiently processing data larger than GPU memory by ' ...
                           'optimally chunking it and distributing higher-dimensional chunks across ' ...
                           'multiple GPUs, while ensuring energy-insulated boundary conditions to ' ...
                           'prevent energy leakage at the edges.\n\nSee https://hydraimageprocessor.com/']);
    opts.ToolboxImageFile = fullfile(repoRoot, 'logo.png');

    % Only the user-facing package ships; tests, perf scripts, and the build
    % scripts stay out of the toolbox (mirrors the .prj exclude list).
    hipDir = fullfile(matlabDir, '+HIP');
    opts.ToolboxFiles = [string(hipDir); string(fullfile(matlabDir, 'CudaMexTester.m'))];
    opts.ToolboxMatlabPath = string(matlabDir);

    % MEX binaries are only guaranteed to load on the MATLAB release they
    % were built with or newer; the workflow builds with this same release.
    opts.MinimumMatlabRelease = 'R2024b';

    opts.SupportedPlatforms.Win64        = true;
    opts.SupportedPlatforms.Glnxa64      = true;
    opts.SupportedPlatforms.Maci64       = false;
    opts.SupportedPlatforms.MatlabOnline = false;

    opts.OutputFile = fullfile(matlabDir, 'Hydra Image Processor.mltbx');

    % Fail loudly if the MEX binaries are missing - a toolbox without them
    % would silently fall back to the CPU implementations for every call.
    mexFiles = {fullfile(hipDir, '@Cuda', 'HIP.mexw64'), fullfile(hipDir, '@Cuda', 'HIP.mexa64')};
    for i = 1:numel(mexFiles)
        if ~exist(mexFiles{i}, 'file')
            error('packageToolboxCI:missingMex', 'Missing MEX binary: %s', mexFiles{i});
        end
    end

    matlab.addons.toolbox.packageToolbox(opts);
    fprintf('Packaged %s (version %s)\n', opts.OutputFile, version);
end
