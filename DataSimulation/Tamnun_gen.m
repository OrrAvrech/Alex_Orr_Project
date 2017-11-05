function finished = Tamnun_gen(samp_num, NumFrames, MaxSources, save_to_dir)
        %%file handling:
    file_ind = get_curr_file_ind(save_to_dir); %returns index of last file created ; creates destination directory if it doesn't exist.\
        %%generate samples, one for each iteration
    parfor i=1:samp_num
        emitters = generate_emitters(MaxSources); %% xyz coordinates
        tetrapod_ans = Data2TetrapodGen( emitters, NumFrames);
        video_samp = tetrapod_ans.LinearCombinations;
        label_samp = label_tetrapods(emitters, MaxSources); %% create labeling images, add zeros(200,200) images to fill number of MaxSources
        dataset_file_handling(video_samp, label_samp, save_to_dir, i+file_ind); %%save combined x,y sample
    end
    finished = 1;
end