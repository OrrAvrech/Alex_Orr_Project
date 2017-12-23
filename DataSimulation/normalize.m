function result = normalize(source)
result = zeros(size(source));
mean_img = sum(mean(source,3));
std_img = std(source,1,3);
for i = 1:size(source,3)
    

%     figure(i); subplot(3,2,1); imagesc(source(:,:,i)); title('before  ');
%     result(:,:,i) = source(:,:,i)./max(source(:,:,i)) * 255; %% max value
% %     is allways 255
%     figure(i); subplot(3,2,2); imagesc(result(:,:,i)); title('after divide by max and mulitply by 255');
%     
%     figure(i); subplot(3,2,3); imagesc(source(:,:,i)); title('before  ');
%     result(:,:,i) = source(:,:,i)./max(source(:,:,i)) * 1; %% max value
% %     is allways 1
%     figure(i); subplot(3,2,4); imagesc(result(:,:,i)); title('after divide by max and mulitply by 1');
% 
%     figure(i); subplot(3,2,5); imagesc(source(:,:,i)); title('before  ');
    result(:,:,i) = (source(:,:,i)-mean_img)./std_img; %% minus mean divide by std
%     figure(i); subplot(3,2,6); imagesc(result(:,:,i)); title('after minus mean divided by std normalization');
end


