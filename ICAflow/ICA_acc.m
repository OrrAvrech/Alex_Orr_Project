function [ error ] = ICA_acc( x_orig, y_orig, ica_res )
    num_of_emitters = size(ica_res,3);
    N =(25e-6).^2 ; %normalization: area of one sample
    temp_err = 0;
    for i = 1:num_of_emitters
        I = mat2gray(ica_res(:,:,i));
        Ibw = im2bw(I);
        Ibw_n = imcomplement(Ibw); % negative
        %Ilabel = bwlabel(Ibw_n);
        Ilabel = logical(Ibw_n); %matlab suggestion to save memory
        stat = regionprops(Ilabel,'centroid');
        imshow(I); hold on; 
        curr_x = stat(1).Centroid(1);
        curr_y  = stat(1).Centroid(2);
        curr_error = 1;
        for ii = 1:numel(x_orig)
            it_dist = sqrt((curr_x - x_orig(ii)).^2+(curr_y - y_orig(ii)).^2);
            it_err = it_dist/N;
            if it_err < curr_error
                curr_error = it_err;
                best_ind = ii; %%may be useful for debug and more
            end
        end
        x_orig(best_ind) = [] ; %delete element
        y_orig(best_ind) = [] ; %delete element
        temp_err = temp_err + curr_error;
        plot(stat(1).Centroid(1),stat(1).Centroid(2),'ro'); %debug
    end
    error = temp_err/num_of_emitters;
end

