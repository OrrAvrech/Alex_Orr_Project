function labels = label_tetrapods(Emitters, MaxSources)
    n = numel(Emitters.x);
    x = Emitters.x;
    y = Emitters.y;
    z = Emitters.zVec;
    labels = zeros(200,200,MaxSources);
    parfor i = 1:n
        xyz = [x(i), y(i), z(i)];
        tetrapod_ans = Data2TetrapodGen(xyz , 1);
        labels(:,:,i) = tetrapod_ans;
    end
end