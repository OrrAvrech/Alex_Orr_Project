function finished = Tamnun_gen(samp_num, NumFrames, MaxSources, save_to_dir)
        %%file handling:
    file_ind = get_curr_file_ind(save_to_dir); %returns index of last file created ; creates destination directory if it doesn't exist.\
    log_hand = create_log(samp_num,NumFrames,MaxSources,save_to_dir,file_ind);
        %%generate samples, one for each iteration
    for i=1:samp_num
        emitters = generate_emitters(MaxSources); %% xyz coordinates
        tetrapod_ans = Data2TetrapodGen( emitters, NumFrames);
        video_samp = tetrapod_ans.LinearCombinations;
        label_samp = label_tetrapods(emitters, MaxSources); %% create labeling images, add zeros(200,200) images to fill number of MaxSources
<<<<<<< HEAD
        video_samp = poissrnd(video_samp) + sigma*randn(size(video_samp)); %%add noise 
        dataset_file_handling(video_samp, label_samp, save_to_dir, i+file_ind); %%save combined x,y sample
=======
%         visualise();
        video_samp = add_noise(video_samp,0); %%add noise 
%         video_samp = normalize(video_samp); %% Normalize
        dataset_file_handling(video_samp, label_samp,emitters , save_to_dir, i+file_ind); %%save combined x,y sample        
>>>>>>> temp-21/11/17
    end
    fprintf( log_hand,'Finished time: %s',datestr(datetime('now')));
    fclose(log_hand);
    finished = 1;
end