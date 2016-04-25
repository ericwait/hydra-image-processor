curVals = curIm(imBW);
curMinVal = min(curVals(:));
curMaxVal = max(curVals(:));
curDiffVal = curMaxVal - curMinVal;

curDelta = 0.02 * curDiffVal;

prevMask = imBW;
imXor = true(size(prevMask));
imH = ImUtils.ThreeD.ShowMaxImage(imXor,true);
ax = get(imH,'Parent');
se = Cuda.MakeBallMask(1);

iter = 0;

imGrow = true(size(prevMask));
imXor = true(size(prevMask));
while (any(imXor(:)) && iter<50)
    imGrow = Cuda.Mex('RegionGrowing',curIm,se,prevMask,curDelta,false);
    imXor = xor(prevMask,imGrow);
    prevMask = imGrow;
    ImUtils.ThreeD.ShowMaxImage(imGrow,false,[],ax,true);
    drawnow
    iter = iter +1;
end
disp(iter);

% imMask = imBW;
% curIm = im(:,:,:,1,1);
% 
% prevMask = imMask;
% imGrow = true(size(prevMask));
% imXor = true(size(prevMask));
% imH = ImUtils.ThreeD.ShowMaxImage(imXor,true);
% ax = get(imH,'Parent');
% se = Cuda.MakeBallMask(1);
% delta = 200;
% iter = 0;
% imGrow = true(size(prevMask));
% imXor = true(size(prevMask));
% while (any(imXor(:)) && iter<5)
%     imGrow = Cuda.Mex('RegionGrowing',curIm,se,prevMask,delta);
%     imXor = xor(prevMask,imGrow);
%     prevMask = imGrow;
%     ImUtils.ThreeD.ShowMaxImage(imGrow,false,[],ax);
%     drawnow
%     iter = iter +1;
% end
% disp(iter);
% 
% delta = 125;
% iter = 0;
% imGrow = true(size(prevMask));
% imXor = true(size(prevMask));
% while (any(imXor(:)) && iter<10)
%     imGrow = Cuda.Mex('RegionGrowing',curIm,se,prevMask,delta);
%     imXor = xor(prevMask,imGrow);
%     prevMask = imGrow;
%     ImUtils.ThreeD.ShowMaxImage(imGrow,false,[],ax);
%     drawnow
%     iter = iter +1;
% end
% disp(iter);
% 
% delta = 75;
% iter = 0;
% imGrow = true(size(prevMask));
% imXor = true(size(prevMask));
% while (any(imXor(:)) && iter<15)
%     imGrow = Cuda.Mex('RegionGrowing',curIm,se,prevMask,delta);
%     imXor = xor(prevMask,imGrow);
%     prevMask = imGrow;
%     ImUtils.ThreeD.ShowMaxImage(imGrow,false,[],ax);
%     drawnow
%     iter = iter +1;
% end
% disp(iter);
% 
% delta = 25;
% iter = 0;
% imGrow = true(size(prevMask));
% imXor = true(size(prevMask));
% while (any(imXor(:)) && iter<100)
%     imGrow = Cuda.Mex('RegionGrowing',curIm,se,prevMask,delta);
%     imXor = xor(prevMask,imGrow);
%     prevMask = imGrow;
%     ImUtils.ThreeD.ShowMaxImage(imGrow,false,[],ax);
%     drawnow
%     iter = iter +1;
% end
% disp(iter);