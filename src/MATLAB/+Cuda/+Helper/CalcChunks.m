function [ localChunks ] = CalcChunks( orgImageDims_rc, deviceDims_rc, kernalDims_rc)
%[ localChunks ] = CalcChunks( orgImageDims, deviceDims, prop, kernalDims, maxThreads)
if (~exist('kernalDims_rc','var') || isempty(kernalDims_rc))
    kernalDims_rc = [0,0,0];
end

margin = round((kernalDims_rc + 1)/2);
chunkDelta = (deviceDims_rc-margin*2);
numChunks_rc = [1,1,1];

if (orgImageDims_rc(1)>deviceDims_rc(1))
    numChunks_rc(1) = ceil(orgImageDims_rc(1)/chunkDelta(1));
else
    chunkDelta(1) = orgImageDims_rc(1);
end

if (orgImageDims_rc(2)>deviceDims_rc(2))
    numChunks_rc(2) = ceil(orgImageDims_rc(2)/chunkDelta(2));
else
    chunkDelta(2) = orgImageDims_rc(2);
end

if (orgImageDims_rc(3)>deviceDims_rc(3))
    numChunks_rc(3) = ceil(orgImageDims_rc(3)/chunkDelta(3));
else
    chunkDelta(3) = orgImageDims_rc(3);
end

localChunks = struct('ImageStart_rc',{[1,1,1]},...
	'ChunkROIstart_rc',{[1,1,1]},...
	'ImageROIstart_rc',{[1,1,1]},...
	'ImageEnd_rc',{[1,1,1]},...
	'ChunkROIend_rc',{[1,1,1]},...
	'ImageROIend_rc',{[1,1,1]});
localChunks(prod(numChunks_rc)).imageStart = [1,1,1];

for curChunk_z=1:numChunks_rc(3)
    for curChunk_c=1:numChunks_rc(2)
        for curChunk_r=1:numChunks_rc(1)
            curChunk_rc = [curChunk_r,curChunk_c,curChunk_z];
            imageROIstart = chunkDelta .* (curChunk_rc-1) + 1;
            imageROIend = min(imageROIstart + chunkDelta, orgImageDims_rc);
            imageStart = max(imageROIstart-margin, [1,1,1]);
            imageEnd = min(imageROIend + margin, orgImageDims_rc);
            chunkROIstart = imageROIstart - imageStart +1;
            chunkROIend = imageROIend - imageROIstart + chunkROIstart +1;
            
            chunkIdx = Utils.CoordToInd(numChunks_rc,curChunk_rc);
            
            localChunks(chunkIdx).ImageStart_rc = imageStart;
            localChunks(chunkIdx).ChunkROIstart_rc = chunkROIstart;
            localChunks(chunkIdx).ImageROIstart_rc = imageROIstart;
            localChunks(chunkIdx).ImageEnd_rc = imageEnd;
            localChunks(chunkIdx).ChunkROIend_rc = chunkROIend;
            localChunks(chunkIdx).ImageROIend_rc = imageROIend;
        end
    end
end
end
