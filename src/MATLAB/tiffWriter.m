function tiffWriter(image,prefix)
sizes = size(image);
numDim = length(sizes);

if numDim<5
    channels = 1;
else
    channels = sizes(5);
end
if numDim<4
    frames = 1;
else
    frames = sizes(4);
end
if numDim<3
    stacks = 1;
else
    stacks = sizes(3);
end


for c=1:channels
    for t=1:frames
        for z=1:stacks
            fileName = sprintf('%s_c%d_t%04d_z%04d.tif',prefix,c,t,z);
            imwrite(image(:,:,z,t,c),fileName,'tif','Compression','lzw');
        end
    end
end

end

