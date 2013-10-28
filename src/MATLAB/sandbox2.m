%% read
im = tiffReader('DAPI Olig2-514 GFAP-488 Dcx-647 Laminin-Cy3 Bcatenin-568 20x20');
chan1_8 = uint8(im(:,:,:,1,1));

%% threshold 
chan1Bin = CudaMex('OtsuThresholdFilter',chan1_8,0.8);
chan1Close = CudaMex('MorphClosure',chan1Bin,[3,3,2]);
chan1Open = CudaMex('MorphOpening',chan1Close,[5,5,2]);

%% draw
figure
imagesc(chan1_8(:,:,20))
colormap gray
figure
imagesc(chan1Bin(:,:,20))
colormap gray
figure
imagesc(chan1Close(:,:,20))
colormap gray
figure
imagesc(chan1Open(:,:,20))
colormap gray