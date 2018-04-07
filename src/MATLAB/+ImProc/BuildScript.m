%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set List of Files to Exclude               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear mex
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
numFunctions = 0;
for i=1:length(dList)
    if (any(strcmpi(excludeList,dList(i).name)))
        continue;
    end

    % copy the file from the class to the package
    newFile = fullfile(packagePath,dList(i).name);
%     if (exist(newFile,'file') && ~strcmpi(dList(i).name,'Help.m') && ~strcmpi(dList(i).name,'Info.m'))
%         % fprintf(1,'File exist: %s\n',newFile);
%         continue;
%     end
%     fprintf(1,'Making undetected file: %s\n',newFile);

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
    funcCallFilled = '';
    for j=1:length(textLines)
        curLine = textLines{j};

        % check for comment line
        if (strcmpi(curLine(1),'%'))
            %check if it is the protype line
            protoLineExpr = '\.Cuda\.';
            protoIdx = regexpi(curLine,protoLineExpr);
            if (~isempty(protoIdx))
                % remove the class part of the path
                fprintf(f, '%s\n',curLine([1:protoIdx,protoIdx+6:end]));
                continue;
            end
            % doesn't me other searches write as is
            fprintf(f,'%s\n',curLine);
            continue;
        end

        % figure out if this is the function line
        funcLineExpr = 'function (?<out>.*) = (?<name>\w+)\((?<param>.*)\)';
        funcCall = regexpi(curLine,funcLineExpr,'names');
        if (~isempty(funcCall))
            funcCallFilled = funcCall;
            fprintf(f,'\nfunction %s = %s(%s)\n',funcCall.out,funcCall.name,funcCall.param);
            fprintf(f,'    try\n');
            fprintf(f,'        %s = ImProc.Cuda.%s(%s);\n',funcCall.out,funcCall.name,funcCall.param);
            fprintf(f,'    catch errMsg\n');
            localFuctionCall = sprintf('%s = ImProc.Local.%s(%s)',funcCall.out,funcCall.name,funcCall.param);
            fprintf(f,'        warning(errMsg.message);\n');
            fprintf(f,'        %s;\n', localFuctionCall);
            fprintf(f,'    end\n');
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
    fclose(f);
    
    numFunctions = numFunctions +1;

    %place the local function call stub here
    if (isempty(funcCallFilled))
        continue
    end
    
    if (~exist(fullfile(packagePath,'+Local'),'dir'))
        mkdir(fullfile(packagePath,'+Local'));
    end
    
    localFuncFileName = fullfile(packagePath,'+Local',[funcCallFilled.name,'.m']);
    if (~exist(localFuncFileName,'file'))
        f = fopen(localFuncFileName,'wt');
        fprintf(f, 'function %s = %s(%s,suppressWarning)\n',funcCallFilled.out,funcCallFilled.name,funcCallFilled.param);
        fprintf(f, '     error(''%s not yet implemented in MATLAB!''); %%delete this line when implemented\n',funcCallFilled.name);
        fprintf(f, '     if (~exist(''suppressWarning'',''var'') || isempty(suppressWarning) || ~suppressWarning)\n');
        fprintf(f, '         warning(''Falling back to matlab.'');\n');
        fprintf(f, '     end\n');
        fprintf(f, '     \n');
        fprintf(f, '     if (~exist(''numIterations'',''var'') || isempty(numIterations))\n');
        fprintf(f, '         numIterations = 1;\n');
        fprintf(f, '     end\n');
        fprintf(f, '     \n');
        fprintf(f, '     arrayOut = arrayIn;\n');
        fprintf(f, '     for t=1:size(arrayIn,5)\n');
        fprintf(f, '         for c=1:size(arrayIn,4)\n');
        fprintf(f, '             for i=1:numIterations\n');
        fprintf(f, '                 %% implement this function here\n');
        fprintf(f, '                 arrayOut(:,:,:,c,t) = arrayIn(:,:,:,c,t);\n');
        fprintf(f, '             end\n');
        fprintf(f, '         end\n');
        fprintf(f, '     end\n');
        fprintf(f, 'end\n');
        fclose(f);
    end
end

fprintf('ImProc BuildScript wrote %d functions\n',numFunctions);
% go back to the original directory
cd(curPath)
