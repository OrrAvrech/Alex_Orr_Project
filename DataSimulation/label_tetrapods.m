function labels = label_tetrapods(Emitters, MaxSources)
    n = numel(Emitters.x);
    x = Emitters.x;
    y = Emitters.y;
    ZposIndex = Emitters.ZposIndex;
    zVec = Emitters.zVec;
    z = zVec(ZposIndex);
    labels = zeros(200,200,MaxSources);
    for i = 1:n
        xyz = [x(i), y(i), z(i)];
        tetrapod_ans = Data2TetrapodGen(xyz , 1);
        labels(:,:,i) = tetrapod_ans;
    end
end