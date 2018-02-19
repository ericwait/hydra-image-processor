%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set List of Files to Exclude               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
excludeList = ...
    {'Cuda.m';
     'DeviceCount.m';
     'DeviceStats.m'};
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
        % fprintf(1,'File exist: %s\n',newFile);
        continue;
    end
    fprintf(1,'Making undetected file: %s\n',newFile);

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
            fprintf(f,'function %s = %s(%s,forceMATLAB)\n',funcCall.out,funcCall.name,funcCall.param);
            fprintf(f, '    if (~exist(''forceMATLAB'',''var'') || isempty(forceMATLAB))\n');
            fprintf(f, '       forceMATLAB = false;\n');
            fprintf(f, '    end\n');
            fprintf(f, '    \n');
            fprintf(f, '    %% check for Cuda capable devices\n');
            fprintf(f, '    [devCount,m] = ImProc.Cuda.DeviceCount();\n');
	        fprintf(f, '    n = length(devCount);\n');
            fprintf(f, '    \n');
            fprintf(f, '    %% if there are devices find the availble one and grab the mutex\n');
            fprintf(f, '    if (n>0 && ~forceMATLAB)\n');
            fprintf(f, '       [~,I] = max([m.available]);\n');
            fprintf(f, '       try\n');
            % call the Cuda version of the function
            fprintf(f, '            %s = ImProc.Cuda.%s(%s,I);\n',funcCall.out,funcCall.name,funcCall.param);
            fprintf(f, '        catch errMsg\n');
            fprintf(f, '        	throw(errMsg);\n');
            fprintf(f, '        end\n');
            fprintf(f, '        \n');
            % otherwise
            fprintf(f, '    else\n');
            localFuctionCall = sprintf('%s = lcl%s(%s)',funcCall.out,funcCall.name,funcCall.param);
            fprintf(f, '        %s;\n', localFuctionCall);
            fprintf(f, '    end\n');
            continue;
        end

        mexCallExpr = '\.Mex';
        mexPos = regexpi(curLine,mexCallExpr);
        if (~isempty(mexPos))
            % this was written above in the Cuda version of the function
            continue
        end

        % does not meet any criteria, write as is
        fprintf(f, '%s\n',curLine);
    end

    %place the local function call stub here
    fprintf(f, '\n');
    fprintf(f, 'function %s\n',localFuctionCall);
    fprintf(f, '\n');
    fprintf(f, 'end\n');
    fprintf(f, '\n');
    fclose(f);
end

% go back to the original directory
cd(curPath)
