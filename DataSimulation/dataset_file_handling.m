function dataset_file_handling(video_samp, label_samp, emitters, save_to, file_ind)
    %%file handling:
    original_path = cdir(save_to);
    
    features = video_samp;
    labels = label_samp;
    positions = emitters;
    save([num2str(file_ind),'.mat'],'-v7.3','features','labels','positions');

    cd (original_path);
end