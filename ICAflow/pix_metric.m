function [ x,y ] = pix_metric( X_orig, Y_orig )
    
    original_path = cdir('..\TetrapodPSF\');
    Sequence = TetrapodGenerator(num2cell(X_orig), num2cell(Y_orig), 1, 1);
    cd (original_path);
    [x,y] = mass_cent(IC);

end