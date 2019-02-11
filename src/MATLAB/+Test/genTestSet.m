function genTestSet(rootDir)
    imageDir = fullfile(rootDir, 'Testing','Images');
    if ( ~exist(imageDir, 'dir') )
        mkdir(imageDir);
    end

    rng_states = [];
    rng(0,'twister');
    
    pxSize = [1,1,3];
    imFullDims = [315, 405, 100, 2, 4];
    
    for ndims = 1:5
        nsdims = min(ndims,3);
        
        rng_states = [rng_states; rng()];
        [imRect, imNoise, imSum] = Test.genImages(imFullDims(1:ndims), pxSize(1:nsdims));
        
        im = imRect;
        save(fullfile(imageDir,['test_image_' num2str(ndims) 'd_rect.mat']), 'im');
        im = imNoise;
        save(fullfile(imageDir,['test_image_' num2str(ndims) 'd_noise.mat']), 'im');
        im = imSum;
        save(fullfile(imageDir,['test_image_' num2str(ndims) 'd_sum.mat']), 'im');
    end
    
    save(fullfile(imageDir, 'rng_states.mat'), 'rng_states');
end
