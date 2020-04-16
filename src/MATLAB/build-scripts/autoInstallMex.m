function autoInstallMex(moduleName, mexFile)
    className = 'Cuda';

    hipDir = fullfile(pwd(),'..',['+' moduleName]);
    if ( ~exist(hipDir,'dir') )
        mkdir(hipDir);
    end

    BuildMexClass(mexFile, hipDir, className, moduleName);
    wrapClassFuncs(hipDir, className);
end

function wrapClassFuncs(packageDir, className)
    classDir = fullfile(packageDir, ['@' className]);

    % Exclude internal/cuda-specific functions
    excludeList = {[className '.m'];
        'DeviceCount.m';
        'DeviceStats.m'};

    % get a list of all of the functions in the class
    funcList = dir(fullfile(classDir,'*.m'));

    for i=1:length(funcList)
        if (any(strcmpi(excludeList, funcList(i).name)))
            continue;
        end

        wrapFile = fullfile(packageDir,funcList(i).name);
        inFile = fullfile(classDir,funcList(i).name);

        inData = fileread(inFile);
        inLines = strsplit(inData, {'\n','\r\n'});

        wrapLines = updateLines(inLines, className);
        wrapStr = strjoin(wrapLines, '\n');
        fid = fopen(wrapFile, 'wt');
        fprintf(fid,'%s', wrapStr);
        fclose(fid);
    end
end

function outLines = updateLines(inLines, className)
    outLines = {};
    for i=1:length(inLines)
        chkLine = inLines{i};

        classExpr = regexptranslate('escape', ['.' className '.']);
        commentProtoExpr = ['(%.*?)' classExpr];
        funcLineExpr = 'function (?<out>.*) = (?<name>\w+)\((?<params>.*)\)';

        commentProto = regexpi(chkLine, commentProtoExpr, 'once');
        funcLine = regexpi(chkLine, funcLineExpr, 'once');
        if ( ~isempty(commentProto) )
            outLines = [outLines; {regexprep(chkLine, commentProtoExpr, '$1.')}];
        elseif ( ~isempty(funcLine) )
            funcCall = regexpi(chkLine, funcLineExpr, 'names');

            outLines = [outLines; {sprintf('function %s = %s(%s)', funcCall.out, funcCall.name, funcCall.params)}];
            outLines = [outLines; {sprintf('    try')}];
            outLines = [outLines; {sprintf('        %s = HIP.Cuda.%s(%s);',funcCall.out,funcCall.name,funcCall.params)}];
            outLines = [outLines; {sprintf('    catch errMsg')}];
            outLines = [outLines; {sprintf('        warning(errMsg.message);')}];
            outLines = [outLines; {sprintf('        %s = HIP.Local.%s(%s);',funcCall.out,funcCall.name,funcCall.params)}];
            outLines = [outLines; {sprintf('    end')}];
            outLines = [outLines; {sprintf('end')}];
            outLines = [outLines; {''}];

            break;
        else
            outLines = [outLines; {chkLine}];
        end
    end
end
