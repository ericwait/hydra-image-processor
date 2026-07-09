classdef SmokeTest < matlab.unittest.TestCase
% Structural checks for the compiled HIP MEX interface that need no GPU.
% CI (matlab-multibuild.yml) runs this on both platforms right after
% building the MEX and regenerating the wrappers; GPU correctness is
% covered by AccuracyTest.m, which must run on a machine with a CUDA
% device (see TESTING.md).

    properties
        HipDir
    end

    methods (TestClassSetup)
        function locateMex(testCase)
            matlabDir = fileparts(fileparts(which('Test.SmokeTest')));
            testCase.HipDir = fullfile(matlabDir, '+HIP');
            mexFile = fullfile(testCase.HipDir, '@Cuda', ['HIP.' mexext]);
            testCase.fatalAssertTrue(isfile(mexFile), sprintf( ...
                ['Compiled MEX missing: %s. Build the HydraMex target with ' ...
                 '-DHYDRA_MODULE_NAME=HIP first.'], mexFile));
        end
    end

    methods (Test)
        function mexLoadsAndListsCommands(testCase)
            info = HIP.Cuda.Info();
            testCase.verifyClass(info, 'struct');
            testCase.verifyNotEmpty(info);
            testCase.verifyTrue(isfield(info, 'command'), ...
                'Info() entries should have a command field');
        end

        function deviceCountRunsWithoutGpu(testCase)
            % Must not error on a machine with no CUDA device (returns 0).
            n = HIP.Cuda.DeviceCount();
            testCase.verifyTrue(isnumeric(n) && isscalar(n), ...
                'DeviceCount should return a numeric scalar');
            testCase.verifyGreaterThanOrEqual(double(n), 0);
        end

        function wrappersMatchCommandTable(testCase)
            % Every command the MEX reports must have a generated class
            % method and, unless excluded by autoInstallMex, a package-level
            % wrapper (the GPU-try/CPU-fallback shims).
            noWrapper = {'Cuda', 'DeviceCount', 'DeviceStats'};
            info = HIP.Cuda.Info();
            for i = 1:numel(info)
                cmd = info(i).command;
                classFile = fullfile(testCase.HipDir, '@Cuda', [cmd '.m']);
                testCase.verifyTrue(isfile(classFile), ...
                    ['Missing generated class method: ' classFile]);
                if ~any(strcmp(noWrapper, cmd))
                    wrapFile = fullfile(testCase.HipDir, [cmd '.m']);
                    testCase.verifyTrue(isfile(wrapFile), ...
                        ['Missing package wrapper: ' wrapFile]);
                end
            end
        end
    end
end
