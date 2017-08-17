function [ ] = Visualize_Sources( seq_lincomb, IC, imagesIC_flag, BlinkMovie_flag )

%%% Generates a blinking movie of mixed sources if BlinkMovie_flag = 1
%%% Generates images of estimated sources by ICA if imagesIC_flag = 1

%% Blinking Movie (ICA input visualization)

if BlinkMovie_flag == 1
    size_lincomb = size(seq_lincomb);
    NumFrames = size_lincomb(3);
    for i=1 : NumFrames
        imagesc(seq_lincomb(:,:,i));
        pause (1) ;
    end
end

%% Estimated Source (ICA output visualization)

if imagesIC_flag == 1
    size_IC = size(IC);
    num_sources = size_IC(3);
    for j=1 : num_sources
        figure(j);
        imagesc(IC(:,:,j));
    end
end
    
end

