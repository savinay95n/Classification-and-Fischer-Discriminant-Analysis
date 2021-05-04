function[weight_matrix, train_pred, test_pred, train_class_mat, test_class_mat] = LeastSquareClassifierFunction(train_feature_matrix, train_labels, test_feature_matrix, test_labels)
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
train_class_mat = train_conf_mat./(meshgrid(countcats(train_labels))');
disp(train_conf_mat);
test_conf_mat = confusionmat(sort(grp2idx(test_labels)),argmaxtest);
disp(test_conf_mat);

test_class_mat = test_conf_mat./(meshgrid(countcats(test_labels))');


return


