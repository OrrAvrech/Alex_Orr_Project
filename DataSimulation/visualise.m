function [outputArg1,outputArg2] = visualise(source1,source2, subplot_num)

    for i = 1:size(source,3)
        figure(i)
        [X2,map2] = imshow('forest.tif');
        subplot(1,2,1), subimage(X)
    end

end

