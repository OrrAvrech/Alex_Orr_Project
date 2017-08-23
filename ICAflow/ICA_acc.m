function [ error ] = ICA_acc( x_orig, y_orig, ica_res )
    num_of_emitters = size(ica_res,3);
    N =25^2 ; %normalization: area of one sample
    temp_err = 0;
    best_ind = [];
    for i = 1:num_of_emitters
        [curr_x,curr_y] = mass_cent(ica_res(:,:,i));
        curr_error = 1;
        
        for ii = 1:numel(x_orig)
            [x,y] = pix_metric(x_orig(ii),y_orig(ii),0);
            it_dist = sqrt((curr_x - x).^2+(curr_y - y).^2);
            it_err = it_dist/N;
            if it_err < curr_error
                curr_error = it_err;
                best_ind = ii; %%may be useful for debug and more
            end
        end
        
        [x,y] = pix_metric(x_orig(best_ind),y_orig(best_ind),1);
        x_orig(best_ind) = [] ; %delete element
        y_orig(best_ind) = [] ; %delete element
        temp_err = temp_err + curr_error;
        
    end
    
    figure(3);plot(curr_x,curr_y,'bo');drawnow; %debug
    title(['ICA image Mass Center Estimation \n Estimated X: ' num2str(curr_x) '\n Estimated Y: ' num2str(curr_y)])
    figure(2);plot(x,y,'ro');drawnow; %debug
    title('Original Mass Center Estimation')
    
    error = temp_err/num_of_emitters;
end

