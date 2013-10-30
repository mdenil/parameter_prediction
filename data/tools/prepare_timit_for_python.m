%% load

clear;

winSzSuffix = '_winSz11'; % _winSz3 or _winSz11 or nothing

% this file has the division into sentences
d = load('TIMIT-MFCCdata_normalized.mat');

%% train

load(['Data_Train3600' winSzSuffix]);
load(['Target_Train3600']);

% move data from columns into rows and 
% remove the extra blank frame from the beginning (wtf is this doin' here)
train_data = single(transpose(Data_Train3600(:,2:end)));
train_labels = single(transpose(Target_Train3600(:,2:end)));

% reconstruct sentence ids
train_sentence_ids = cell(length(d.traindata),1);
for i = 1:length(d.traindata)
    train_sentence_ids{i} = i + zeros(size(d.traindata{i},1),1);
end
train_sentence_ids = uint32(cell2mat(train_sentence_ids));
train_sentence_ids = train_sentence_ids - 1; % move to zero indexing

% sanity checks
assert(all(all(train_labels == cell2mat(d.trainlab'))));

% too much memory :\
clear Data_Train3600
clear Target_Train3600

%% validation

load(['Data_Dev' winSzSuffix]);
load(['Target_Dev']);

valid_data = single(transpose(Data_Dev(:,2:end)));
valid_labels = single(transpose(Target_Dev(:,2:end)));

valid_sentence_ids = cell(length(d.devsetdata),1);
for i = 1:length(d.devsetdata)
    valid_sentence_ids{i} = i + zeros(size(d.devsetdata{i},1),1);
end
valid_sentence_ids = uint32(cell2mat(valid_sentence_ids));
valid_sentence_ids = valid_sentence_ids - 1;

assert(all(all(valid_labels == cell2mat(d.devsetlab'))));

clear Data_Dev
clear Target_Dev

%% test

load(['Data_Test' winSzSuffix]);
load(['Target_Test']);

test_data = single(transpose(Data_Test(:,2:end)));
test_labels = single(transpose(Target_Test(:,2:end)));

test_sentence_ids = cell(length(d.coretestdata),1);
for i = 1:length(d.coretestdata)
    test_sentence_ids{i} = i + zeros(size(d.coretestdata{i},1),1);
end
test_sentence_ids = uint32(cell2mat(test_sentence_ids));
test_sentence_ids = test_sentence_ids - 1;

assert(all(all(test_labels == cell2mat(d.coretestlab'))));

clear Data_Test
clear Target_Test

%%

clear d;

%% save train data in batches because fuck mat files

n_batches = 5;

batch_sz = ceil(size(train_data,1)/5);

from = 1;
for i = 1:n_batches
    fname = ['timit_train_b' num2str(i)];
    disp(['Saving ' fname '...']);
    
    to = min(from + batch_sz - 1, size(train_data,1));
    
    X = train_data(from:to,:);
    Y = train_labels(from:to,:);
    sentence_ids = train_sentence_ids(from:to);
    save(['for_python/' fname], 'X', 'Y', 'sentence_ids');
    clear X;
    clear Y;
    
    from = from + batch_sz;
end
assert( to == size(train_data,1) );

%%

X = valid_data;
Y = valid_labels;
sentence_ids = valid_sentence_ids;
save('for_python/timit_valid.mat', 'X', 'Y', 'sentence_ids');

%%

X = test_data;
Y = test_labels;
sentence_ids = test_sentence_ids;
save('for_python/timit_test.mat', 'X', 'Y', 'sentence_ids');
