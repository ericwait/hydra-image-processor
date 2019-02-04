function compareSnapshots(baseDir, snapDir)
    baseList = dir(fullfile(baseDir,'*.mat'));
    snapList = dir(fullfile(snapDir,'*.mat'));
    
    baseFiles = {baseList.name}.';
    snapFiles = {snapList.name}.';
    
    missingBase = setdiff(snapFiles,baseFiles);
    missingSnap = setdiff(baseFiles,snapFiles);
    
    commonFiles = intersect(baseFiles,snapFiles);
    
    for i=1:length(missingBase)
        fprintf('++++++++: %s\n', missingBase{i});
    end
    
    for i=1:length(missingSnap)
        fprintf('--------: %s\n', missingSnap{i});
    end
    
    for i=1:length(commonFiles)
        base = load(fullfile(baseDir,commonFiles{i}));
        if ( isfield(base,'imOut') )
            imField = 'imOut';
        elseif ( isfield(base,'im') )
            imField = 'im';
        else
            warning(['No image data found in ' commonFiles{i}]);
            continue;
        end
        
        snap = load(fullfile(snapDir,commonFiles{i}));
        if ( ~isfield(snap,imField) )
            warning(['Image data not shared between baseline/snapshot: ' commonFiles{i}])
            continue;
        end
        
        imChk = max(base.(imField));
        imRange = [min(imChk(:)), max(imChk(:))];
        imDiffSq = (base.(imField) - snap.(imField)).^2;
        if ( sqrt(mean(imDiffSq)) > (1e-6)*(imRange(2)-imRange(1)) )
            fprintf('ERROR_HI: %s (%f)\n', commonFiles{i}, sqrt(mean(imDiffSq(:))));
        else
            fprintf('        : %s (%f)\n', commonFiles{i}, sqrt(mean(imDiffSq(:))));
        end
    end
end
