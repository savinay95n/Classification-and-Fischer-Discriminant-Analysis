% Main Program to train and test Fisher Projected data points using
% Generative Model Classifier.
%% Wine Dataset
clc;
clear all;
close all;

% Feature subspace dimension
% Can range from 1 to D, where D is the number of dimansions.
m = 2;

% training using 2 features for convenience of visualization

% Please enter the features you want to train and test on
featureA = 1;
featureB = 2;

% Loading wine dataset
[train_features,train_labels,test_features,test_labels] = loadDataset('wine');

% Getting test and train Projected feature Matrices
[Y_train, WeightVector_train] = fisherLDA(train_features, train_labels,m);
[Y_test, WeightVector_test] = fisherLDA(test_features, test_labels,m);

N_total_test = length(test_labels);
N_total_train = length(train_labels);
numGroups = length(unique(test_labels));
category_names = categories(test_labels);

% Plotting
colors = jet(numGroups*10);
colors = colors(round(linspace(1,numGroups*10,numGroups)),:);
figure(1);
h1 = gscatter(Y_test(:,featureA),Y_test(:,featureB),test_labels,'','+o*v^');
for i = 1:numGroups
    h1(i).LineWidth = 2;
    h1(i).MarkerEdgeColor = min(colors(i,:)*1.2,1);
end
lim_info =  [min(Y_test(:,featureA)),max(Y_test(:,featureA)),...
    min(Y_test(:,featureB)),max(Y_test(:,featureB))  ];


X = Y_test(:,[featureA,featureB]);

% Linear Discriminant for projected test datausing decision theory
covar = 0;
for i = 1 : numGroups
    classIndices{i} = find(grp2idx(test_labels) == i);
    N(i) = length(classIndices{i});
    prior(i) = N(i)/N_total_test;
    mu{i} = mean(X(classIndices{i},:));
    for n = classIndices{i}(1) : classIndices{i}(end)
        covar = covar + (1/(N_total_test - numGroups)) * (X(classIndices{i},:) - mu{i})' * (X(classIndices{i},:) - mu{i});
    end
end
hold on;



x = [min(X(:,featureA)),max(X(:,featureA))];

for i = 1 : numGroups
    for j = 1 : numGroups
        if i ~= j
            [FisherLd.Coeffs(i,j).slope,FisherLd.Coeffs(i,j).intercept] = fisherlinearDisc(prior(i),prior(j),mu{i},mu{j},covar);
            y = FisherLd.Coeffs(i,j).intercept + FisherLd.Coeffs(i,j).slope * x;
            plot(x,y, 'LineWidth',2,'DisplayName',sprintf('Class Sep b/w %s,%s',category_names{i},category_names{j}));
            title('{\bf Linear Discriminant Classification}')

        end
    end
end
axis(lim_info)
hold off
grid on;
set(gca,'FontWeight','bold','LineWidth',2)

% Prior and Covariance Matrix calculation for test dataset
covar_test = 0;
for i = 1 : numGroups
    classIndices_test{i} = find(grp2idx(test_labels) == i);
    N_test(i) = length(classIndices{i});
    prior_test(i) = N_test(i)/N_total_test;
    mu_test{i} = mean(Y_test(classIndices_test{i},:));
    for n = classIndices_test{i}(1) : classIndices_test{i}(end)
        covar_test = covar_test + (1/(N_total_test - numGroups)) * (Y_test(classIndices_test{i},:) - mu_test{i})' * (Y_test(classIndices_test{i},:) - mu_test{i});
    end
end

% Prior and Covariance Matrix calculation for train dataset
covar_train = 0;
for i = 1 : numGroups
    classIndices_train{i} = find(grp2idx(train_labels) == i);
    N_train(i) = length(classIndices_train{i});
    prior_train(i) = N_train(i)/N_total_train;
    mu_train{i} = mean(Y_train(classIndices_train{i},:));
    for n = classIndices_train{i}(1) : classIndices_train{i}(end)
        covar_train = covar_train + (1/(N_total_train - numGroups)) * (Y_train(classIndices_train{i},:) - mu_train{i})' * (Y_train(classIndices_train{i},:) - mu_train{i});
    end
end

% Prediction of labels for test and train data using Generative Probabilistic Model
[prediction_train,confmat_train] = fisherPredict(Y_train,train_labels,prior,mu_train,covar_train);
[prediction_test,confmat_test] = fisherPredict(Y_test,test_labels,prior_test,mu_test,covar_test);

classmat_train = confmat_train./(meshgrid(countcats(train_labels))');
classmat_test = confmat_test./(meshgrid(countcats(test_labels))');

train_acc = mean(diag(classmat_train))
train_std = std(diag(classmat_train));
test_acc = mean(diag(classmat_test))
test_std = std(diag(classmat_test));


%% Wallpaper Dataset
clc;
clear all;
close all;
% Feature subspace dimension
% Can range from 1 to D, where D is the number of dimansions.
m = 16;

% Please enter the features you want to train and test on
featureA = 1;
featureB = 2;

% Loading Wallpaper Dataset
[train_features,train_labels,test_features,test_labels] = loadDataset('wallpaper');

% Getting test and train Projected feature Matrices
[Y_train, WeightVector_train] = fisherLDA(train_features, train_labels,m);
[Y_test, WeightVector_test] = fisherLDA(test_features, test_labels,m);

N_total_test = length(test_labels);
N_total_train = length(train_labels);
numGroups = length(unique(test_labels));
category_names = categories(test_labels);

% Plotting
colors = jet(numGroups*10);
colors = colors(round(linspace(1,numGroups*10,numGroups)),:);
figure(1);
h1 = gscatter(Y_test(:,featureA),Y_test(:,featureB),test_labels,'','+o*v^');
for i = 1:numGroups
    h1(i).LineWidth = 2;
    h1(i).MarkerEdgeColor = min(colors(i,:)*1.2,1);
end
lim_info =  [min(Y_test(:,featureA)),max(Y_test(:,featureA)),...
    min(Y_test(:,featureB)),max(Y_test(:,featureB))  ];


X = Y_test(:,[featureA,featureB]);

% Linear Discriminant for projected test datausing decision theory
covar = 0;
for i = 1 : numGroups
    classIndices{i} = find(grp2idx(test_labels) == i);
    N(i) = length(classIndices{i});
    prior(i) = N(i)/N_total_test;
    mu{i} = mean(X(classIndices{i},:));
    for n = classIndices{i}(1) : classIndices{i}(end)
        covar = covar + (1/(N_total_test - numGroups)) * (X(classIndices{i},:) - mu{i})' * (X(classIndices{i},:) - mu{i});
    end
end
hold on;



x = [min(X(:,featureA)),max(X(:,featureA))];

for i = 1 : numGroups
    for j = 1 : numGroups
        if i ~= j
            [FisherLd.Coeffs(i,j).slope,FisherLd.Coeffs(i,j).intercept] = fisherlinearDisc(prior(i),prior(j),mu{i},mu{j},covar);
            y = FisherLd.Coeffs(i,j).intercept + FisherLd.Coeffs(i,j).slope * x;
            plot(x,y, 'LineWidth',2,'DisplayName',sprintf('Class Sep b/w %s,%s',category_names{i},category_names{j}));
            title('{\bf Linear Discriminant Classification}')

        end
    end
end
axis(lim_info)
hold off
grid on;
set(gca,'FontWeight','bold','LineWidth',2)

% Prior and Covariance Matrix calculation for test dataset
covar_test = 0;
for i = 1 : numGroups
    classIndices_test{i} = find(grp2idx(test_labels) == i);
    N_test(i) = length(classIndices{i});
    prior_test(i) = N_test(i)/N_total_test;
    mu_test{i} = mean(Y_test(classIndices_test{i},:));
    for n = classIndices_test{i}(1) : classIndices_test{i}(end)
        covar_test = covar_test + (1/(N_total_test - numGroups)) * (Y_test(classIndices_test{i},:) - mu_test{i})' * (Y_test(classIndices_test{i},:) - mu_test{i});
    end
end

% Prior and Covariance Matrix calculation for train dataset
covar_train = 0;
for i = 1 : numGroups
    classIndices_train{i} = find(grp2idx(train_labels) == i);
    N_train(i) = length(classIndices_train{i});
    prior_train(i) = N_train(i)/N_total_train;
    mu_train{i} = mean(Y_train(classIndices_train{i},:));
    for n = classIndices_train{i}(1) : classIndices_train{i}(end)
        covar_train = covar_train + (1/(N_total_train - numGroups)) * (Y_train(classIndices_train{i},:) - mu_train{i})' * (Y_train(classIndices_train{i},:) - mu_train{i});
    end
end

% Prediction of labels for test and train data using Generative Probabilistic Model
[prediction_train,confmat_train] = fisherPredict(Y_train,train_labels,prior,mu_train,covar_train);
[prediction_test,confmat_test] = fisherPredict(Y_test,test_labels,prior_test,mu_test,covar_test);

classmat_train = confmat_train./(meshgrid(countcats(train_labels))');
classmat_test = confmat_test./(meshgrid(countcats(test_labels))');

train_acc = mean(diag(classmat_train))
train_std = std(diag(classmat_train));
test_acc = mean(diag(classmat_test))
test_std = std(diag(classmat_test));

%% Taiji Dataset
clc;
clear all;
close all;

% Feature subspace dimension
% Can range from 1 to D, where D is the number of dimansions.
m = 7;

% Please enter the features you want to train and test on
featureA = 1;
featureB = 2;

% Loading Taiji dataset
[train_features,train_labels,test_features,test_labels] = loadDataset('taiji');

% Getting test and train Projected feature Matrices
[Y_train, WeightVector_train] = fisherLDA(train_features, train_labels,m);
[Y_test, WeightVector_test] = fisherLDA(test_features, test_labels,m);

N_total_test = length(test_labels);
N_total_train = length(train_labels);
numGroups = length(unique(test_labels));
category_names = categories(test_labels);

% Plotting
colors = jet(numGroups*10);
colors = colors(round(linspace(1,numGroups*10,numGroups)),:);
figure(1);
h1 = gscatter(Y_test(:,featureA),Y_test(:,featureB),test_labels,'','+o*v^');
for i = 1:numGroups
    h1(i).LineWidth = 2;
    h1(i).MarkerEdgeColor = min(colors(i,:)*1.2,1);
end
lim_info =  [min(Y_test(:,featureA)),max(Y_test(:,featureA)),...
    min(Y_test(:,featureB)),max(Y_test(:,featureB))  ];


X = Y_test(:,[featureA,featureB]);

% Linear Discriminant for projected test data using decision theory
covar = 0;
for i = 1 : numGroups
    classIndices{i} = find(grp2idx(test_labels) == i);
    N(i) = length(classIndices{i});
    prior(i) = N(i)/N_total_test;
    mu{i} = mean(X(classIndices{i},:));
    for n = classIndices{i}(1) : classIndices{i}(end)
        covar = covar + (1/(N_total_test - numGroups)) * (X(classIndices{i},:) - mu{i})' * (X(classIndices{i},:) - mu{i});
    end
end
hold on;



x = [min(X(:,featureA)),max(X(:,featureA))];

for i = 1 : numGroups
    for j = 1 : numGroups
        if i ~= j
            [FisherLd.Coeffs(i,j).slope,FisherLd.Coeffs(i,j).intercept] = fisherlinearDisc(prior(i),prior(j),mu{i},mu{j},covar);
            y = FisherLd.Coeffs(i,j).intercept + FisherLd.Coeffs(i,j).slope * x;
            plot(x,y, 'LineWidth',2,'DisplayName',sprintf('Class Sep b/w %s,%s',category_names{i},category_names{j}));
            title('{\bf Linear Discriminant Classification}')

        end
    end
end
axis(lim_info)
hold off
grid on;
set(gca,'FontWeight','bold','LineWidth',2)

% Prior and Covariance Matrix calculation for test dataset
covar_test = 0;
for i = 1 : numGroups
    classIndices_test{i} = find(grp2idx(test_labels) == i);
    N_test(i) = length(classIndices{i});
    prior_test(i) = N_test(i)/N_total_test;
    mu_test{i} = mean(Y_test(classIndices_test{i},:));
    for n = classIndices_test{i}(1) : classIndices_test{i}(end)
        covar_test = covar_test + (1/(N_total_test - numGroups)) * (Y_test(classIndices_test{i},:) - mu_test{i})' * (Y_test(classIndices_test{i},:) - mu_test{i});
    end
end

% Prior and Covariance Matrix calculation for train dataset
covar_train = 0;
for i = 1 : numGroups
    classIndices_train{i} = find(grp2idx(train_labels) == i);
    N_train(i) = length(classIndices_train{i});
    prior_train(i) = N_train(i)/N_total_train;
    mu_train{i} = mean(Y_train(classIndices_train{i},:));
    for n = classIndices_train{i}(1) : classIndices_train{i}(end)
        covar_train = covar_train + (1/(N_total_train - numGroups)) * (Y_train(classIndices_train{i},:) - mu_train{i})' * (Y_train(classIndices_train{i},:) - mu_train{i});
    end
end

% Prediction of labels for test and train data using Generative Probabilistic Model
[prediction_train,confmat_train] = fisherPredictTaiji(Y_train,train_labels,prior,mu_train,covar_train);
[prediction_test,confmat_test] = fisherPredictTaiji(Y_test,test_labels,prior_test,mu_test,covar_test);

classmat_train = confmat_train./(meshgrid(countcats(train_labels))');
classmat_test = confmat_test./(meshgrid(countcats(test_labels))');

train_acc = mean(diag(classmat_train))
train_std = std(diag(classmat_train));
test_acc = mean(diag(classmat_test))
test_std = std(diag(classmat_test));

