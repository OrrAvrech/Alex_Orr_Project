function log_handler = create_log(samp_num,NumFrames,MaxSources,save_to_dir,file_ind)
    fid = fopen(fullfile(save_to_dir,strcat('log--',strrep(strrep(datestr(datetime('now')),':','-'),' ','--'),'.txt')), 'wt');
    fprintf( fid, 'Start Time: %s\nCreating %d samples, first file number is: %d\nWith %d frames \nMax number of sources is: %d\n', datestr(datetime('now')), samp_num,file_ind+1, NumFrames, MaxSources);
    log_handler = fid;
end
