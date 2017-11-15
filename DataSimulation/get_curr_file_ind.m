function file_ind = get_curr_file_ind(save_to_dir,file_ind)
    file_naming_ind = 0;
            %%make sure the folder exists
    orig_listing = dir; % dir(name) returns file list containg name sub string
    orig_listing = {orig_listing.name};
    exists = find(ismember(orig_listing,save_to_dir));
    if ~numel(exists)
        mkdir(save_to_dir);
    end
    
    original_path = cdir([save_to_dir,'\']); %change to saving destination
    
    dest_listing = dir;
    if ~dest_listing(end).isdir %is empty?
        dest_names = ({dest_listing.name});
        dest_names = cellfun(@(x) str2num(x(1:end-4)), dest_names, 'un', 0);
%         last_file = dest_names{end};
%         file_naming_ind = str2num(last_file(1:end-4)) %remove '.mat'
        file_naming_ind = max([dest_names{:}])
    end
    
    
%     dest_listing = dir
%     if ~dest_listing(end).isdir %is empty?
%         dest_names = ({dest_listing.name})
%         last_file = dest_names{end};
%         file_naming_ind = str2num(last_file(1:end-4)) %remove '.mat'
%     end
    file_ind = file_naming_ind ;
    cd(original_path)
end