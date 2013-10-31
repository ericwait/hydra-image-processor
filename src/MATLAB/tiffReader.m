function im = tiffReader(datasetName,chan,type,dirName,nameConvention,cPos,tPos,zPos)
global DatasetName OrgDir

DatasetName = datasetName;
im = [];

if (~exist('dirName','var') || isempty(dirName))
    fprintf('Select Image Dir...');
    dirName = uigetdir();
    disp(dirName);
end

if (~exist('nameConvention','var') || isempty(nameConvention))
    nameConvention = '_c%d_t%04d_z%04d%s';
    cPos = 1;
    tPos = 2;
    zPos = 3;
end

if (~exist('chan','var')|| isempty(chan))
    chan = 0;
end

if (~exist('type','var') || isempty(type))
    type = 'double';
end

if (strcmp(type,'double'))
    bytes = 8;
elseif (strcmp(type,'uint8'))
    bytes = 1;
else
    error('Type not implemented');
end

OrgDir = dirName;

dList = dir(fullfile(dirName,'*.tif*'));

mxC = 0;
mxT = 0;
mxZ = 0;

mxX = 0;
mxY = 0;

for i=1:length(dList)
    vals = textscan(dList(i).name,[datasetName nameConvention]);
    
    if (isempty(vals) || isempty(vals{1}) || isempty(vals{2}) || isempty(vals{3}))
        continue;
    end
    
    if (mxX==0 || mxY==0)
        im = imread(fullfile(dirName,dList(i).name));
        mxX = size(im,2);
        mxY = size(im,1);
    end
    
    c = double(vals{cPos});
    t = double(vals{tPos});
    z = double(vals{zPos});
    
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

if (chan~=0)
    mxC=1;
end

fprintf('%d,%d,%d,%d,%d, %4.2fGB\n',mxY,mxX,mxZ,mxT,mxC,mxY*mxX*mxZ*mxT*mxC*bytes/1024/1024/1024);

im = zeros(1,1,1,1,1);

if (bytes==8)
    im = zeros(mxY,mxX,mxZ,mxT,mxC);
elseif (bytes==1)
    im = zeros(mxY,mxX,mxZ,mxT,mxC,'uint8');
end

for i=1:length(dList)
    vals = textscan(dList(i).name,[datasetName nameConvention]);
    
    if (isempty(vals) || isempty(vals{1}) || isempty(vals{2}) || isempty(vals{3}))
        continue;
    end
    
    c = vals{cPos};
    if (chan~=0 && c~=chan)
        continue;
    end
    
    t = vals{tPos};
    z = vals{zPos};
   
    if (bytes==8)
        im(:,:,z,t,c) = imread(fullfile(dirName,dList(i).name));
    elseif (bytes==1)
        im(:,:,z,t,c) = uint8(imread(fullfile(dirName,dList(i).name)));
    end
end
