figure

plot(0,0,'.w');
ax = gca;
hold on

for i=1:length(imChunks)
    curChunk = imChunks(i);
    
    Utils.PlotBox(ax,Utils.SwapXY_RC(curChunk.ImageStart_rc),Utils.SwapXY_RC(curChunk.ImageEnd_rc),'-b',num2str(i));
    Utils.PlotBox(ax,Utils.SwapXY_RC(curChunk.ImageROIstart_rc),Utils.SwapXY_RC(curChunk.ImageROIend_rc),'--r');
    Utils.PlotBox(ax,...
        Utils.SwapXY_RC(curChunk.ImageStart_rc) + Utils.SwapXY_RC(curChunk.ChunkROIstart_rc) - 1,...
        Utils.SwapXY_RC(curChunk.ImageStart_rc) + Utils.SwapXY_RC(curChunk.ChunkROIstart_rc) -1 + (Utils.SwapXY_RC(curChunk.ChunkROIend_rc)-Utils.SwapXY_RC(curChunk.ChunkROIstart_rc)-1),...
        ':g');
end
axis ij
axis equal
