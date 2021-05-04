%% LeastSquareClassifierFunction() 
% *************************************************************************
%--------------------------------------------------------------------------
% This function trains a least square classifier on
% any given data set by finding a closed form solution for the optimal
% Weight Vector W. This W is then used on a test data set to predict the
% category/label, each data point in it belongs to. The target matrix is
% obtained from the function binaryCodedTarget(). Train and Test
% Classification Matrices are found from the predicted vectors to determine
% accuracy of clasification. 
% This function is used for the main Program LeastSquaresFisher.m, to show
% the relationship between Fisher Criterion and Least Squares.
%--------------------------------------------------------------------------

% Inputs: Training Feature Matrix | [N1 x D]
%         Training labels | [N1 x 1]
%         Testing Feature Matrix | [N2 x D]
%         Testing labels | [N2 x 1]

% Ouputs: Weight Matrix | [D+1 x K]
%         Train Prediction Vector | [N1 x 1]
%         Test Prediction Vector | [N2 x K]
%         Train Confusion Matrix | [K x K]
%         Test Confusion Matrix | [K x K]

% Written by: Savinay Nagendra (sxn265@psu.edu)
% *************************************************************************

%% Function
function[weight_matrix, train_pred, test_pred,train_conf_mat,test_conf_mat] = LeastSquareClassifierFunction2(train_feature_matrix, train_labels, test_feature_matrix, test_labels)
[train_target,sorted_train_feature_matrix] = binaryCodedTarget(train_labels,train_feature_matrix);
[test_target,sorted_test_feature_matrix] = binaryCodedTarget(test_labels,test_feature_matrix);
[num_categories,~,~] = unique(train_labels);
train_feature_matrix_tilda = [ones(length(train_labels),1),sorted_train_feature_matrix];
test_feature_matrix_tilda = [ones(length(test_labels),1),sorted_test_feature_matrix];
[~,num_features] = size(sorted_train_feature_matrix);
weight_matrix = zeros(num_features+1 , length(num_categories));

weight_matrix  = (train_feature_matrix_tilda'*train_feature_matrix_tilda)\(train_feature_matrix_tilda'*train_target);

train_prediction_matrix = train_feature_matrix_tilda * weight_matrix;
test_prediction_matrix = test_feature_matrix_tilda * weight_matrix;

for i = 1:length(train_prediction_matrix)
    [~,argmaxtrain(i)] = max(train_prediction_matrix(i,:));
end
train_pred = argmaxtrain';
for i = 1:length(test_prediction_matrix)
    [~,argmaxtest(i)] = max(test_prediction_matrix(i,:));
end
test_pred = argmaxtest';

train_conf_mat = confusionmat(sort(grp2idx(train_labels)),argmaxtrain);
test_conf_mat = confusionmat(sort(grp2idx(test_labels)),argmaxtest);


return


