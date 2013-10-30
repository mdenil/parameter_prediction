%%

clear;

load TIMIT-MFCC/TIMIT-MFCCdata_normalized.mat

%%

for i = 1:length(trainlab)
    if any(trainlab{i}(:,61:63))
        break;
    end
end


%%

phone_files = phone_file_list();

label_types = {'rise', 'mid', 'fall'};


label_index = cell(size(trainlab{1},2),1);

found_all = false;

for file_idx = 1:length(phone_files)
    
    fid = fopen(phone_files{file_idx}, 'r');
    phone_data = textscan(fid, '%s %s %s');
    fclose(fid);
    phone_data = phone_data{3};
    
    % make sure the phone_data file we loaded has the same number of phones
    % as the corresponding trainlab labels.
    if (length(phone_data) ~= ceil(sum(sum(diff(trainlab{file_idx}, 1)~=0, 1)/2)/3))
        warning(['Wrong number of phones']);
        continue
    end
    
    prev_label_idx = 0;
    phone_idx = 0;
    for time_idx = 1:size(trainlab{file_idx}, 1)
        frame = trainlab{file_idx}(time_idx, :);
        label_idx = find(frame);
        assert(length(label_idx) == 1);
        
        % silly 1-based indexing
        label_type = mod(label_idx - 1, 3)+1;
        label_idx = (label_idx - 1) - (label_type - 1) + 1;
        
        if prev_label_idx ~= label_idx
            phone_idx = phone_idx + 1;
            prev_label_idx = label_idx;
        end
        
        label_string = [phone_data{phone_idx} ' ' label_types{label_type}];
        if numel(label_index{label_idx + (label_type - 1)}) == 0
            label_index{label_idx + (label_type - 1)} = label_string;
        else
            suspected_label = label_index{label_idx + (label_type - 1)};
            if length(suspected_label) ~= length(label_string) || ~(all(suspected_label == label_string))
                warning([num2str(file_idx) ': ' suspected_label ' != ' label_string]);
                break;
            end
        end
    end
    
    
    if all(cellfun(@numel, label_index)~=0)
        found_all = true;
        break;
    end
    
end

if ~found_all
    warning('Some of the phone labels are still missing');
end

fid = fopen('phone_mapping.txt', 'w');
for i = 1:length(label_index)
    fprintf(fid, '%d, %s', i, label_index{i});
end
