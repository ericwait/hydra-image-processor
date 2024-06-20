function [im_cv, experiments] = CoefficientVariationFilter(im, meta, expected_cell_diam, varargin)
% COEFFICIENTVARIATIONIMAGE Generate an image of the coefficient of variation.
%
% This function calculates the coefficient of variation (CV) of an image. 
% The CV is defined as the standard deviation divided by the mean. 
% The function uses filters to calculate the local standard deviation 
% and mean, then divides the standard deviation by the mean to obtain the CV image.
%
% INPUTS:
%   im                  - The input image (numeric array).
%   meta                - Metadata for the image (struct).
%   expected_cell_diam  - Expected cell diameter (numeric).
%   varargin            - Additional optional parameters (experiments, out_dir, overwrite).
%
% OUTPUTS:
%   im_cv       - The coefficient of variation image (numeric array).
%   experiments - Updated experiments structure (struct).
%
% Optional Parameters:
%   'experiments' (default: empty struct) - An experiments structure.
%   'out_dir' (default: empty string)     - Output directory for saving the CV image.
%   'overwrite' (default: false)          - Flag indicating whether to overwrite existing files.

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'experiments', [], @isstruct);
    addParameter(p, 'out_dir', '', @isstring);
    addParameter(p, 'overwrite', false, @islogical);
    parse(p, varargin{:});
    args = p.Results;

    % Validate inputs
    validateattributes(im, {'numeric'}, {'nonempty'}, mfilename, 'im');
    validateattributes(meta, {'struct'}, {'nonempty'}, mfilename, 'meta');
    validateattributes(expected_cell_diam, {'numeric'}, {'scalar', 'positive'}, mfilename, 'expected_cell_diam');

    % Initialize experiments if provided
    if ~isempty(args.experiments)
        experiments = args.experiments;
        experiments = experiments.setOperationName('cv image');
        meta.DatasetName = [meta.DatasetName, '_var'];

        % Check if the operation needs to run
        if ~experiments.needsRun() && ~args.overwrite
            im_cv = MicroscopeData.Reader(imageData=meta, outType='single');
            return;
        end
    end

    % Convert image to single precision
    im = ImUtils.ConvertType(im, 'single', true);

    % Calculate the entropy size and structuring element
    entropy_size = expected_cell_diam / 0.8;
    se = HIP.MakeEllipsoidMask(floor(entropy_size / meta.PixelPhysicalSize));

    % Apply standard deviation and mean filters
    im_std = HIP.StdFilter(im, se, 1, []);
    im_mean = HIP.MeanFilter(im, se, 1, []);

    % Calculate the coefficient of variation
    im_cv = im_std ./ im_mean;

    % Save the CV image if output directory is specified
    if ~isempty(args.out_dir)
        MicroscopeData.WriterTif(ImUtils.ConvertType(im_cv, 'uint16'), args.out_dir, imageData=meta);
    end

    % Update experiments with parameters if provided
    if ~isempty(args.experiments)
        var_param.entropy_size = entropy_size;
        var_param.struct_el_size = [floor(entropy_size / meta.PixelPhysicalSize(1:2)), 0];
        experiments = experiments.setFieldValue(var_param);
    end
end
