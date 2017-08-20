function [ error ] = ICA_acc( x_orig, y_orig, ica_res )
    num_of_emitters = size(ica_res,3);
    N =25^2 ; %normalization: area of one sample
    temp_err = 0;
    best_ind = [];
    for i = 1:num_of_emitters
        [curr_x,curr_y] = mass_cent(ica_res(:,:,i));
        curr_error = 1;
        
        for ii = 1:numel(x_orig)
            [x,y] = pix_metric(x_orig(ii),y_orig(ii));
            it_dist = sqrt((curr_x - x).^2+(curr_y - y).^2);
            it_err = it_dist/N;
            if it_err < curr_error
                curr_error = it_err;
                best_ind = ii; %%may be useful for debug and more
            end
        end
        curr_x
        x_orig(best_ind)
        curr_y
        y_orig(best_ind)
        x_orig(best_ind) = [] ; %delete element
        y_orig(best_ind) = [] ; %delete element
        temp_err = temp_err + curr_error;
        
    end
    error = temp_err/num_of_emitters;
end

