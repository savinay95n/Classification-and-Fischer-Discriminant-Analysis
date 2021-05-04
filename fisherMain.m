clc;
clear;

[train_features,train_labels,test_features,test_labels] = loadDataset('wine');
m = 2;
featureA = 1;
featureB = 2;
[ Y_train, W_train] = fisherLDA( train_features, train_labels, m);
[ Y_test, W_test] = fisherLDA( test_features, test_labels, m);

category_names = unique(train_labels);
numGroups = length(category_names);
% Plotting 
colors = jet(numGroups*10);
colors = colors(round(linspace(1,numGroups*10,numGroups)),:);

h1 = gscatter(Y_test(:,featureA),Y_test(:,featureB),test_labels,'','+o*v^');
for i = 1:numGroups
    h1(i).LineWidth = 2;
    h1(i).MarkerEdgeColor = min(colors(i,:)*1.2,1);
end

mdl = fitcdiscr(Y_train,train_labels);
tr_pred = predict(mdl,Y_train);
tst_pred = predict(mdl,Y_test);
