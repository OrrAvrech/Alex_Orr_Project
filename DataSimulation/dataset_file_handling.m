function dataset_file_handling(video_samp, label_samp, emitters, save_to, file_ind)
    %%file handling:
    original_path = cdir(save_to);
    
    x = video_samp;
    y = label_samp;
<<<<<<< HEAD
    save([num2str(file_ind),'.mat'],'-v7.3','x', 'y');
=======
%     emitters = emitters;
    save([num2str(file_ind),'.mat'],'-v7.3','x', 'y', 'emitters');
>>>>>>> temp-21/11/17

    cd (original_path);
end