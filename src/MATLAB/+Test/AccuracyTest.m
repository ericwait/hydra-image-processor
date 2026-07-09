classdef AccuracyTest < matlab.unittest.TestCase
    
    properties
        TestDataDir
        Im
        ImNorm
        Kernel
        Sigmas
    end
    
    methods(TestClassSetup)
        function setupData(testCase)
            % Check for GPU availability
            try
                devCount = Hydra.DeviceCount();
                if isstruct(devCount) || iscell(devCount)
                    % Sometimes it returns stats or tuple, handle loosely
                     % In Python wrapper it returns (count, stats) or count
                     % Let's assume if it runs, we check count
                     if iscell(devCount)
                        devCount = devCount{1};
                     end
                end
                
                % If devCount is struct/stats, maybe count is length?
                % Let's rely on documentation/header: SCR_CMD_NOPROC(DeviceCount, SCR_PARAMS(SCR_OUTPUT(SCR_SCALAR(int32_t), numCudaDevices), SCR_OUTPUT(SCR_STRUCT, memStats)))
                % Mex returns [num, stats].
            catch
                devCount = 0;
            end
            
            testCase.assumeTrue(isnumeric(devCount) && devCount > 0, ...
                'No CUDA devices found. Skipping accuracy tests.');

            % Locate test data relative to this file
            % File is in src/MATLAB/+Test/AccuracyTest.m
            % Data is in test_data/ (root)
            
            % Get path of this file
            currentFile = mfilename('fullpath');
            [testDir, ~, ~] = fileparts(currentFile);
            % Go up from +Test -> MATLAB -> src -> root -> test_data
            testCase.TestDataDir = fullfile(testDir, '..', '..', '..', 'test_data');
            
            testCase.assumeTrue(isfolder(testCase.TestDataDir), ...
                ['Test data directory not found at: ' testCase.TestDataDir]);
            
            im0Path = fullfile(testCase.TestDataDir, 'test_c0.tif');
            im1Path = fullfile(testCase.TestDataDir, 'test_c1.tif');
            
            testCase.assumeTrue(isfile(im0Path) && isfile(im1Path), ...
                'Test source images missing');
                
            im0 = MicroscopeData.LoadTif(im0Path);
            im1 = MicroscopeData.LoadTif(im1Path);
            testCase.Im = cat(4, im0, im1);
            testCase.ImNorm = ImUtils.ConvertType(testCase.Im, 'single', true);
            
            testCase.Sigmas = [19, 19, 9];
            radius = [5, 5, 2];
            testCase.Kernel = HIP.MakeEllipsoidMask(radius);
        end
    end
    
    methods
        function verifyImage(testCase, actual, expectedFile, msg)
            expectedPath = fullfile(testCase.TestDataDir, expectedFile);
            if ~isfile(expectedPath)
                % Soft warning/skip if ground truth missing, or fail?
                % Standard unit tests usually fail if resources missing unless assumed.
                % We will assume it exists for "Test" to pass, but print warning if not?
                % Let's use assumption so it marks as 'Incomplete' if missing, not Failed.
                testCase.assumeTrue(isfile(expectedPath), ['Missing ground truth: ' expectedFile]);
                return;
            end
            
            expected = MicroscopeData.LoadTif(expectedPath);
            
            if contains(expectedFile, '_c1_')
                actualChannel = actual(:,:,:,1);
            elseif contains(expectedFile, '_c2_')
                actualChannel = actual(:,:,:,2);
            else
                actualChannel = actual;
            end
            
            testCase.verifyEqual(size(actualChannel), size(expected), [msg ' (Shape)']);
            
            if isfloat(actualChannel)
                diff = abs(actualChannel - expected);
                maxDiff = max(diff(:));
                testCase.verifyLessThan(maxDiff, 1e-4, [msg ' (Value mismatch)']);
            else
                testCase.verifyEqual(actualChannel, expected, [msg ' (Value mismatch)']);
            end
        end
    end
    
    methods(Test)
        function testGaussian(testCase)
            imOut = Hydra.Gaussian(testCase.Im, testCase.Sigmas, 1, []);
            testCase.verifyImage(imOut, 'Gaussian_c1_19-19-9.tif', 'Gaussian c1');
            testCase.verifyImage(imOut, 'Gaussian_c2_19-19-9.tif', 'Gaussian c2');
        end
        
        function testHighPassFilter(testCase)
            imOut = Hydra.HighPassFilter(testCase.Im, testCase.Sigmas, []);
            testCase.verifyImage(imOut, 'HighPassFilter_c1_19-19-9.tif', 'HighPassFilter c1');
            testCase.verifyImage(imOut, 'HighPassFilter_c2_19-19-9.tif', 'HighPassFilter c2');
        end
        
        function testLoG(testCase)
            sigmas_log = [4, 4, 2];
            imOut = Hydra.LoG(testCase.ImNorm, sigmas_log, []);
            imOut(imOut > 0.001) = 0;
            imOut = abs(imOut);
            testCase.verifyImage(imOut, 'LoG_c1_4-4-2.tif', 'LoG c1');
            testCase.verifyImage(imOut, 'LoG_c2_4-4-2.tif', 'LoG c2');
        end
        
        function testClosure(testCase)
            imOut = Hydra.Closure(testCase.Im, testCase.Kernel, 1, []);
            testCase.verifyImage(imOut, 'Closure_c1_5-5-2.tif', 'Closure c1');
            testCase.verifyImage(imOut, 'Closure_c2_5-5-2.tif', 'Closure c2');
        end
        
        function testMaxFilter(testCase)
            imOut = Hydra.MaxFilter(testCase.Im, testCase.Kernel, 1, []);
            testCase.verifyImage(imOut, 'MaxFilter_c1_5-5-2.tif', 'MaxFilter c1');
            testCase.verifyImage(imOut, 'MaxFilter_c2_5-5-2.tif', 'MaxFilter c2');
        end
        
        function testMeanFilter(testCase)
            imOut = Hydra.MeanFilter(testCase.Im, testCase.Kernel, 1, []);
            testCase.verifyImage(imOut, 'MeanFilter_c1_5-5-2.tif', 'MeanFilter c1');
            testCase.verifyImage(imOut, 'MeanFilter_c2_5-5-2.tif', 'MeanFilter c2');
        end
        
        function testMedianFilter(testCase)
            imOut = Hydra.MedianFilter(testCase.Im, testCase.Kernel, 1, []);
            testCase.verifyImage(imOut, 'MedianFilter_c1_5-5-2.tif', 'MedianFilter c1');
            testCase.verifyImage(imOut, 'MedianFilter_c2_5-5-2.tif', 'MedianFilter c2');
        end
        
        function testMinFilter(testCase)
            imOut = Hydra.MinFilter(testCase.Im, testCase.Kernel, 1, []);
            testCase.verifyImage(imOut, 'MinFilter_c1_5-5-2.tif', 'MinFilter c1');
            testCase.verifyImage(imOut, 'MinFilter_c2_5-5-2.tif', 'MinFilter c2');
        end
        
        function testOpener(testCase)
            imOut = Hydra.Opener(testCase.Im, testCase.Kernel, 1, []);
            testCase.verifyImage(imOut, 'Opener_c1_5-5-2.tif', 'Opener c1');
            testCase.verifyImage(imOut, 'Opener_c2_5-5-2.tif', 'Opener c2');
        end
        
        function testStdFilter(testCase)
            imOut = Hydra.StdFilter(testCase.Im, testCase.Kernel, 1, []);
            testCase.verifyImage(imOut, 'StdFilter_c1_5-5-2.tif', 'StdFilter c1');
            testCase.verifyImage(imOut, 'StdFilter_c2_5-5-2.tif', 'StdFilter c2');
        end
        
        function testVarFilter(testCase)
            imOut = Hydra.VarFilter(testCase.Im, testCase.Kernel, 1, []);
            testCase.verifyImage(imOut, 'VarFilter_c1_5-5-2.tif', 'VarFilter c1');
            testCase.verifyImage(imOut, 'VarFilter_c2_5-5-2.tif', 'VarFilter c2');
        end
        
        function testMultiplySum(testCase)
            imOut = Hydra.MultiplySum(testCase.Im, testCase.Kernel.*3, 1, []);
            testCase.verifyImage(imOut, 'MultiplySum_c1_15-15-6.tif', 'MultiplySum c1');
            testCase.verifyImage(imOut, 'MultiplySum_c2_15-15-6.tif', 'MultiplySum c2');
        end
        
        function testElementWiseDifference(testCase)
            imOut = Hydra.ElementWiseDifference(testCase.Im, testCase.Im(:,:,:,end:-1:1), []);
            testCase.verifyImage(imOut, 'ElementWiseDifference_c1_reverse.tif', 'ElementWiseDifference c1');
            testCase.verifyImage(imOut, 'ElementWiseDifference_c2_reverse.tif', 'ElementWiseDifference c2');
        end
    end
end
