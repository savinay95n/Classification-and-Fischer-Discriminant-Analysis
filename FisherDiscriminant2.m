clc;
clear all;
m = 13;

featureA = 1;
featureB = 7;

[train_features,train_labels,test_features,test_labels] = loadDataset('wine');


% Plotting
colors = jet(numGroups*10);
colors = colors(round(linspace(1,numGroups*10,numGroups)),:);

h1 = gscatter(Y(:,featureA),Y(:,featureB),train_labels,'','+o*v^');
for i = 1:numGroups
    h1(i).LineWidth = 2;
    h1(i).MarkerEdgeColor = min(colors(i,:)*1.2,1);
end