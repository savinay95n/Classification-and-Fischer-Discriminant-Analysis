clc; 
close all;
clear all;
addpath export_fig
% Choose which dataset to use (choices wine, wallpaper, taiji) :
dataset = 'wallpaper';
[train_featureVector, train_labels,...
    test_featureVector,test_labels]     = loadDataset(dataset);

numClasses = length(countcats(test_labels));

train_labels = grp2idx(train_labels);
[val idx] = sort(train_labels,'ascend');
train_labels = val;
train_featureVector = train_featureVector(idx(:),:);

test_labels = grp2idx(test_labels);
[val idx] = sort(test_labels,'ascend');
test_labels = val;
test_featureVector = test_featureVector(idx(:),:);

feature_idx = 1:size(train_featureVector,2);

%% Fisher Projection
%Training set
Fisher_W = fisher_proj(train_featureVector,...
                         categorical(train_labels), numClasses);
                   
% Fisher_W = Fischer_anand(train_featureVector,...
%                          categorical(train_labels));

Fisher_train_featureVector = train_featureVector*Fisher_W;  
Fisher_train_labels = train_labels;
%Testing set


Fisher_W = fisher_proj(test_featureVector,...
                         categorical(test_labels), numClasses);
                     
% Fisher_W = Fischer_anand(test_featureVector,...
%                          categorical(test_labels));
Fisher_test_featureVector = test_featureVector*Fisher_W;
Fisher_test_labels = test_labels;

% table2 = zeros(9,2);
% for k=1:1:15
k=7;
pred_label = knnClassifier_anand(Fisher_train_featureVector,Fisher_train_labels,...
            Fisher_test_featureVector,k);

% pred_label = knnClassifier(Fisher_train_featureVector,Fisher_train_labels,...
%             Fisher_test_featureVector,k);

% pred_label = categorical(pred_label);

test_ConfMat = confusionmat(test_labels,pred_label);
% Create classification matrix (rows should sum to 1)
test_ClassMat = test_ConfMat./(meshgrid(countcats(categorical(test_labels)))');
% mean group accuracy and std
accuracy_test = mean(diag(test_ClassMat))
test_std = std(diag(test_ClassMat))
 