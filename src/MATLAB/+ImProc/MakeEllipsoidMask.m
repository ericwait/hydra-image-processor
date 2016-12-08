function shapeElement = MakeEllipsoidMask(axesRadius)
    if (numel(axesRadius)~=3)
        error('There must be a value for each axis X, Y, & Z even if it is 0');
    end
    
    volSize = max(ceil(axesRadius*2)+1,ones(size(axesRadius)));
	nbSE = false(volSize);
	[XX,YY,ZZ] = ndgrid(1:volSize(1),1:volSize(2),1:volSize(3));

    % shift origin to the center
    XX = XX-size(nbSE,1)/2-0.5;
    YY = YY-size(nbSE,2)/2-0.5;
    ZZ = ZZ-size(nbSE,3)/2-0.5;
    
    axesRadius = max(axesRadius,1e-10);
    
	bInside = (...
        XX.^2./axesRadius(1)^2 +...
        YY.^2./axesRadius(2)^2 +...
        ZZ.^2./axesRadius(3)^2);
    
    bInsideMask = bInside <=1;
	nbSE(bInsideMask) = true;

	shapeElement = nbSE;
end