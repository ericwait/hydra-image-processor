function im = tiffReader(datasetName,dirName,nameConvention,cPos,tPos,zPos)
global DatasetName OrgDir

DatasetName = datasetName;
im = [];

if (~exist('dirName','var'))
    fprintf('Select Image Dir...');
    dirName = uigetdir();
    disp(dirName);
end

if (~exist('nameConvention','var'))
    nameConvention = '_c%d_t%04d_z%04d%s';
    cPos = 1;
    tPos = 2;
    zPos = 3;
end

OrgDir = dirName;

dList = dir(fullfile(dirName,'*.tif*'));

mxC = 0;
mxT = 0;
mxZ = 0;

for i=1:length(dList)
    vals = textscan(dList(i).name,[datasetName nameConvention]);
    
    if (isempty(vals) || isempty(vals{1}) || isempty(vals{2}) || isempty(vals{3}))
        continue;
    end
    
    c = vals{cPos};
    t = vals{tPos};
    z = vals{zPos};
    im(:,:,z,t,c) = imread(fullfile(dirName,dList(i).name));
    
    if (c>mxC)
        mxC = c;
    end
    if (t>mxT)
        mxT = t;
    end
    if (z>mxZ)
        mxZ = z;
    end
end
