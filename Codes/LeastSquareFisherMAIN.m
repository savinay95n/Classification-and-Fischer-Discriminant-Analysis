% This program is used to show empirically that Fisher's Criterion is a
% special case of Least Squares Method in case of 2 classes.
clc;
clear all;
close all;
% Select the features
featureA = 1;
featureB = 2;
f = [featureA featureB];
[tr_feat,tr_labels,tst_feat,tst_labels] = loadDataset('wine');
classes = categories(tr_labels);
numGroups = length(classes);
% Structure to extract data corresponding for the two features
for i = 1:numGroups
    classIndices_tr{i} = find(grp2idx(tr_labels) == i);
    classIndices_tst{i} = find(grp2idx(tst_labels) == i);

    X.train{i} = tr_feat(classIndices_tr{i},f);
    X.test{i} = tst_feat(classIndices_tst{i},f);
    X.tr_labels{i} = tr_labels(classIndices_tr{i},:);
    X.tst_labels{i} = tst_labels(classIndices_tst{i},:);
end
% You can change the to any two of the three classes in wine 
class1 = 1;
class2 = 2;
% Structure to store data for the two classes selected
data.tr = [X.train{class1};X.train{class2}];
data.tst = [X.test{class1};X.test{class2}];
data.tr_labels = [X.tr_labels{class1};X.tr_labels{class2}];
data.tst_labels = [X.tst_labels{class1};X.tst_labels{class2}];

% Least Squares
[w,tr_pred,tst_pred,tr_conf_mat,tst_conf_mat] = LeastSquareClassifierFunction2(data.tr,data.tr_labels,data.tst,data.tst_labels);

category_names = categories(data.tst_labels);
% number of categories in a label vector
numGroups = length(category_names);
colors = jet(numGroups*10);
colors = colors(round(linspace(1,numGroups*10,numGroups)),:);

%Plotting the data points
h1 = gscatter(data.tst(:,featureA),data.tst(:,featureB),data.tst_labels,'','+o*v^');

lim_info =  [min(data.tst(:,featureA)),max(data.tst(:,featureA)),...
    min(data.tst(:,featureB)),max(data.tst(:,featureB))  ];
hold on
x =  [min(data.tst(:,featureA)),max(data.tst(:,featureA))];
y = (w(1,1) - w(1,2) + (w(2,1) - w(2,2))*x)/(w(3,2) - w(3,1)) ;
plot(x,y,'LineWidth',2);

% Fisher
[Y_tr,W_tr] = fisherLDA(data.tr,data.tr_labels,2);
[Y_tst,W_tst] = fisherLDA(data.tst,data.tst_labels,2);

% Least Squares on Projected data points
[wf,tr_predf,tst_predf,tr_conf_matf,tst_conf_matf] = LeastSquareClassifierFunction2(Y_tr,data.tr_labels,Y_tst,data.tst_labels);
y = (wf(1,1) - wf(1,2) + (wf(2,2) - wf(2,1))*x)/(w(3,2) - w(3,1)) ;
figure(2);
x =  [min(Y_tst(:,featureA)),max(Y_tst(:,featureA))];
h1 = gscatter(Y_tst(:,featureA),Y_tst(:,featureB),data.tst_labels,'','+o*v^');

lim_info =  [min(Y_tst(:,featureA)),max(Y_tst(:,featureA)),...
    min(Y_tst(:,featureB)),max(Y_tst(:,featureB))  ];
hold on
plot(x,y,'LineWidth',2);


