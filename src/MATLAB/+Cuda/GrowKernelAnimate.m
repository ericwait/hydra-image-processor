imMask = imMedFill;
curIm = imS(:,:,:,1,1);

prevMask = imMask;
imGrow = true(size(prevMask));
imXor = true(size(prevMask));
imH = ImUtils.ThreeD.ShowMaxImage(imXor,true);
ax = get(imH,'Parent');
se = Cuda.MakeBallMask(1);
delta = 0.005;
iter = 0;
while (any(imXor(:)) && iter<100)
    imGrow = Cuda.Mex('RegionGrowing',curIm,se,prevMask,delta);
    imXor = xor(prevMask,imGrow);
    prevMask = imGrow;
    ImUtils.ThreeD.ShowMaxImage(imGrow,false,[],ax);
    drawnow
    iter = iter +1;
end
