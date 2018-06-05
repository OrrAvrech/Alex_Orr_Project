function file_ind = get_curr_file_ind(save_to_dir,file_ind)
    file_naming_ind = 0;
            %%make sure the folder exists
    orig_listing = dir; % dir(name) returns file list containg name sub string
    orig_listing = {orig_listing.name};
    exists = find(ismember(orig_listing,save_to_dir));
    if ~numel(exists)
        mkdir(save_to_dir);
    end
    
    original_path = cdir([save_to_dir,'/']); %change to saving destination
    
    dest_listing = dir;
    if ~dest_listing(end).isdir %is empty?
        dest_names = ({dest_listing.name});
        dest_names = dest_names(~cellfun(@isempty,regexp(dest_names,'.mat'))); % filter files that don't end with .mat
        if (numel(dest_names) == 0)
            file_naming_ind = 0;
        else
            last_file = dest_names{end};
            file_naming_ind = str2num(last_file(1:end-4)); %remove '.mat'
        end
    end
%     if (numel(file_naming_ind) == 0)
%         file_naming_ind = 0;
%     end
    file_ind = file_naming_ind ;
    cd(original_path)
end