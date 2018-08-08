function BuildMexObject(mexFile, objectName, parentPackage)
    oldPath = pwd();
    cleanupObj = onCleanup(@()(cleanupFunc(oldPath)));

    [mexPath,mexName] = fileparts(mexFile);
    cd(mexPath);

    mexFunc = str2func(mexName);
	commandList = mexFunc('Info');

    cd(oldPath);
    makeClassdef(objectName, mexName, commandList);
    for i=1:length(commandList)
        makeStaticMethod(objectName, mexName, commandList(i), parentPackage);
    end

    copyfile(mexFile, ['@' objectName]);
    clear mex
end

function makeClassdef(objectName, mexName, commandList)
    if ( ~exist(['@' objectName],'dir') )
        mkdir(['@' objectName]);
    end
    objFile = fopen(fullfile(['@' objectName],[objectName '.m']), 'wt');

    fprintf(objFile, 'classdef (Abstract,Sealed) %s\n', objectName);

    fprintf(objFile, 'methods (Static)\n');
    for i=1:length(commandList)
        fprintf(objFile, '    %s\n', makePrototypeString(commandList(i)));
    end
    fprintf(objFile, 'end\n');

    fprintf(objFile, 'methods (Static, Access = private)\n');
    fprintf(objFile, '    varargout = %s(command, varargin)\n', mexName);
    fprintf(objFile, 'end\n');

    fprintf(objFile, 'end\n');

    fclose(objFile);
end

function makeStaticMethod(objectName, mexName, commandInfo, parentPackage)
    methodFile = fopen(fullfile(['@' objectName],[commandInfo.command '.m']), 'wt');

    summaryString = '';
    if ( length(commandInfo.helpLines) < 2 )
        summaryString = [makePrototypeString(commandInfo) ' '];
    end

    if ( ~isempty(commandInfo.helpLines) )
        summaryString = [summaryString commandInfo.helpLines{1}];
    end

    fprintf(methodFile, '%% %s - %s\n', commandInfo.command, summaryString);

    if ( length(commandInfo.helpLines) > 1 )
        fprintf(methodFile, '%%    %s\n', makePrototypeString(commandInfo,objectName,parentPackage,true));
    end

    for i=2:length(commandInfo.helpLines)
        fprintf(methodFile, '%%    %s\n', commandInfo.helpLines{i});
    end

    fprintf(methodFile, 'function %s\n', makePrototypeString(commandInfo));
    fprintf(methodFile, '    %s;\n', makeCommandString(objectName, mexName,commandInfo,parentPackage));
    fprintf(methodFile, 'end\n');

    fclose(methodFile);
end

function commandString = makeCommandString(objectName, mexName, commandInfo, parentPackage)
    commandString = '';
    if ( ~isempty(commandInfo.outArgs) )
         commandString = ['[' makeCommaList(commandInfo.outArgs) '] = '];
    end
    if ( ~exist('parentPackage','var') )
        parentPackage = [];
    else
        parentPackage = [parentPackage '.'];
    end

    mexCall = [parentPackage objectName '.' mexName];
    commandString = [commandString mexCall '(''' commandInfo.command ''''];
    if ( ~isempty(commandInfo.inArgs) )
         commandString = [commandString ',' makeCommaList(commandInfo.inArgs)];
    end
    commandString = [commandString ')'];
end

function protoString = makePrototypeString(commandInfo, objectName, parentPackage, leaveOptBrackets)
    if ( ~exist('objectName','var') )
        objectName = [];
    end
    if ( ~exist('parentPackage','var') )
        parentPackage = [];
    else
        parentPackage = [parentPackage '.'];
    end
    if ( ~exist('leaveOptBrackets','var') )
        leaveOptBrackets = false;
    end

    protoString = '';
    if ( ~isempty(commandInfo.outArgs) )
        protoString = makeCommaList(commandInfo.outArgs,leaveOptBrackets);

        if ( length(commandInfo.outArgs) > 1 )
            protoString = ['[' protoString ']'];
        end
        protoString = [protoString ' = '];
    end

    if ( ~isempty(objectName) )
        protoString = [protoString parentPackage objectName '.'];
    end

    protoString = [protoString commandInfo.command '('];
    if ( ~isempty(commandInfo.inArgs) )
         protoString = [protoString makeCommaList(commandInfo.inArgs, leaveOptBrackets)];
    end
    protoString = [protoString ')'];
end

function commaStr = makeCommaList(inList, leaveOptBrackets)
    if (~exist('leaveOptBrackets','var') || isempty(leaveOptBrackets))
        leaveOptBrackets = false;
    end
    
    commaStr = '';
    if ( isempty(inList) )
        return;
    end

    for i=1:length(inList)-1
        commaStr = [commaStr removeOptBrackets(inList{i}, leaveOptBrackets) ','];
    end

    commaStr = [commaStr removeOptBrackets(inList{end}, leaveOptBrackets)];
end

function argName = removeOptBrackets(argName, leaveOptBrackets)
    if ( ~leaveOptBrackets )
        argName = regexprep(argName, '\[(\w+)\]', '$1');
    end
end

function cleanupFunc(oldPath)
    cd(oldPath);
    clear mex;
end
