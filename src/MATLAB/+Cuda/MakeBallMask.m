function shapeElement = MakeBallMask(radius)
nbSE = false(2*radius+1,2*radius+1,2*radius+1);
[XX,YY,ZZ] = ndgrid(1:size(nbSE,1),1:size(nbSE,2),1:size(nbSE,2));

bInside = (((XX(:)-radius-1).^2 + (YY(:)-radius-1).^2 + (ZZ(:)-radius-1).^2) <= radius^2);
nbSE(bInside) = true;

shapeElement = nbSE;
end