function emitters = generate_emitters(MaxSources)


        %%inputs
        
        %%choose:
    NumZplanes = 80;
    x_center = 0;
    y_center = 0;
    x_width = 10e-6;
    y_width = 10e-6;
    
        %%helpers:  
    zVec = linspace(-2e-6,2e-6,NumZplanes+1);

        %%stochastic:
    NumSources = randi(MaxSources-1)+1;
        %%actual data:
    x = (rand(1,NumSources)+x_center-0.5)*x_width/2;
    y = (rand(1,NumSources)+y_center-0.5)*y_width/2;
    z = (randi(NumZplanes,[1,NumSources]));
    emitters.x = x;
    emitters.y = y;
    emitters.zVec = zVec;
    emitters.ZposIndex = z;
end