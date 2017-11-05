function dataset_file_handling(video_samp, label_samp, save_to, file_ind)
    %%file handling:
    original_path = cdir(save_to);
    
    dat.x = video_samp;
    dat.y = label_samp;
    save([num2str(file_ind),'.mat'],'-v7.3','dat');

    cd (original_path);
end