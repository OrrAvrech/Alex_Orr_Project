function [ x,y ] = pix_metric( X_orig, Y_orig,display )
    
    original_path = cdir('..\TetrapodPSF\');
    Sequence = TetrapodGenerator(num2cell(X_orig), num2cell(Y_orig), 1, 1);
    if display == 1
        figure(2); imshow(mat2gray(Sequence.LinearCombinations(:,:,1))); hold on;
    end
    cd (original_path);
    [x,y] = mass_cent(Sequence.LinearCombinations);

end