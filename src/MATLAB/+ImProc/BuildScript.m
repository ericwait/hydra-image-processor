%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set List of Files to Exclude               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
excludeList = ...
    {'Cuda.m';
     'DeviceCount.m'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% remember curent path
curPath = pwd();

% find where the image processing package is
cudaPath = fileparts(which('ImProc.BuildMexObject'));
cd(cudaPath)

% create the m files that correspond to the commands in the mex interface
ImProc.BuildMexObject('..\..\c\Mex.mexw64','Cuda','ImProc');

packagePath = cudaPath;
cudaPath = fullfile(cudaPath,'@Cuda');

% get a list of all of the functions in the class
dList = dir(fullfile(cudaPath,'*.m'));

% wrap each function
for i=1:length(dList)
    if (any(strcmpi(excludeList,dList(i).name)))
        continue;
    end

    % copy the file from the class to the package
    newFile = fullfile(packagePath,dList(i).name);
    if (exist(newFile,'file'))
        fprintf(1,'File exist: %s\n',newFile);
        continue;
    end

    % get all of the lines
    f = fopen(fullfile(cudaPath,dList(i).name),'rt');
    curLine = fgetl(f);
    textLines = {};
    while ischar(curLine)
        textLines = [textLines;{curLine}];
        curLine = fgetl(f);
    end
    fclose(f);

    % write out the new data
    f = fopen(newFile,'wt');
    for j=1:length(textLines)
        curLine = textLines{j};

        % check for comment line
        if (strcmpi(curLine(1),'%'))
            %check if it is the protype line
            protoLineExpr = '\.Cuda\.';
            protoIdx = regexpi(curLine,protoLineExpr);
            if (~isempty(protoIdx))
                % remove the class part of the path
                fprintf(f, '%s);\n',curLine([1:protoIdx,protoIdx+6:end-8]));
                continue;
            end

            if (~isempty(regexpi(curLine,'Device --')))
                % remove line
                continue;
            end

            % doesn't me other searches write as is
            fprintf(f,'%s\n',curLine);
            continue;
        end

        % figure out if this is the function line
        funcLineExpr = 'function (?<out>.*) = (?<name>\w+)\((?<param>.*),device\)';
        funcCall = regexpi(curLine,funcLineExpr,'names');
        if (~isempty(funcCall))
            fprintf(f,'function %s = %s(%s)\n',funcCall.out,funcCall.name,funcCall.param);

            % add the device checking
            fprintf(f, '    curPath = which(''ImProc.Cuda'');\n');
            fprintf(f, '    curPath = fileparts(curPath);\n');
            fprintf(f, '    n = ImProc.Cuda.DeviceCount();\n');
            fprintf(f, '    foundDevice = false;\n');
            fprintf(f, '    device = -1;\n');
            fprintf(f, '    \n');
            fprintf(f, '    while(~foundDevice)\n');
            fprintf(f, '    	for deviceIdx=1:n\n');
            fprintf(f, '    		mutexfile = fullfile(curPath,sprintf(''device%%02d.txt'',deviceIdx));\n');
            fprintf(f, '    		if (~exist(mutexfile,''file''))\n');
            fprintf(f, '    			try\n');
            fprintf(f, '                    fclose(fopen(mutexfile,''wt''));\n');
            fprintf(f, '    			catch errMsg\n');
            fprintf(f, '                    continue;\n');
            fprintf(f, '    			end\n');
            fprintf(f, '    			foundDevice = true;\n');
            fprintf(f, '    			device = deviceIdx;\n');
            fprintf(f, '    			break;\n');
            fprintf(f, '    		end\n');
            fprintf(f, '    	end\n');
            fprintf(f, '    	if (~foundDevice)\n');
            fprintf(f, '    		pause(2);\n');
            fprintf(f, '    	end\n');
            fprintf(f, '    end\n');
            fprintf(f, '    \n');
            fprintf(f, '    try\n');
            
            % call the Cuda version of the function
            fprintf(f, '        %s = ImProc.Cuda.%s(%s,device);\n',funcCall.out,funcCall.name,funcCall.param);
            
            % clean up the mutex
            fprintf(f, '    catch errMsg\n');
            fprintf(f, '    	delete(mutexfile);\n');
            fprintf(f, '    	throw(errMsg);\n');
            fprintf(f, '    end\n');
            fprintf(f, '    \n');
            fprintf(f, '    delete(mutexfile);\n');
            continue;
        end

        mexCallExpr = '\.Mex';
        mexPos = regexpi(curLine,mexCallExpr);
        if (~isempty(mexPos))
            % this was written above in the Cuda version of the function
            continue
        end

        % does not meet any criteria, write as is
        fprintf(f,'%s\n',curLine);
    end
    fclose(f);
end

% go back to the original directory
cd(curPath)
