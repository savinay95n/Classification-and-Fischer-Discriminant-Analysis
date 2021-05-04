%% Wine Dataset
clc;
clear all;
m = 2;

featureA = 1;
featureB = 2;

[train_features,train_labels,test_features,test_labels] = loadDataset('wine');


[Y_train, WeightVector_train] = fisherLDA(train_features, train_labels,m);
[Y_test, WeightVector_test] = fisherLDA(test_features, test_labels,m);

N_total = length(test_labels);
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

% Linear Discriminant using decision theory
covar = 0;
for i = 1 : numGroups
    classIndices{i} = find(grp2idx(test_labels) == i);
    N(i) = length(classIndices{i});
    prior(i) = N(i)/N_total;
    mu{i} = mean(X(classIndices{i},:));
    for n = classIndices{i}(1) : classIndices{i}(end)
        covar = covar + (1/(N_total - numGroups)) * (X(classIndices{i},:) - mu{i})' * (X(classIndices{i},:) - mu{i});
    end
end
hold on;

x = [min(X(:,featureA)),max(X(:,featureA))];

for i = 1 : numGroups
    for j = 1 : numGroups
        if i ~= j
            [FisherLd.Coeffs(i,j).slope,FisherLd.Coeffs(i,j).intercept] = fisherlinearDisc(prior(i),prior(j),mu{i},mu{j},covar);
            y = FisherLd.Coeffs(i,j).intercept+ FisherLd.Coeffs(i,j).slope * x;
            plot(x,y, 'LineWidth',2,'DisplayName',sprintf('Class Sep b/w %s,%s',category_names{i},category_names{j}));
            title('{\bf Linear Discriminant Classification}')

        end
    end
end
axis(lim_info)
hold off
grid on;
set(gca,'FontWeight','bold','LineWidth',2)


%% Wallpaper Dataset
clc;
clear all;
m = 2;

featureA = 1;
featureB = 2;

[train_features,train_labels,test_features,test_labels] = loadDataset('wallpaper');


[Y_train, WeightVector_train] = fisherLDA(train_features, train_labels,m);
[Y_test, WeightVector_test] = fisherLDA(test_features, test_labels,m);

N_total = length(test_labels);
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

% Linear Discriminant using decision theory
covar = 0;
for i = 1 : numGroups
    classIndices{i} = find(grp2idx(test_labels) == i);
    N(i) = length(classIndices{i});
    prior(i) = N(i)/N_total;
    mu{i} = mean(X(classIndices{i},:));
    for n = classIndices{i}(1) : classIndices{i}(end)
        covar = covar + (1/(N_total - numGroups)) * (X(classIndices{i},:) - mu{i})' * (X(classIndices{i},:) - mu{i});
    end
end
hold on;
x = [min(X(:,featureA)),max(X(:,featureA))];

for i = 1 : numGroups
    for j = 1 : numGroups
        if i ~= j
            [FisherLd.Coeffs(i,j).slope,FisherLd.Coeffs(i,j).intercept] = fisherlinearDisc(prior(i),prior(j),mu{i},mu{j},covar);
            y = FisherLd.Coeffs(i,j).intercept+ FisherLd.Coeffs(i,j).slope * x;
            plot(x,y, 'LineWidth',2,'DisplayName',sprintf('Class Sep b/w %s,%s',category_names{i},category_names{j}));
            title('{\bf Linear Discriminant Classification}')

        end
    end
end
axis(lim_info)
hold off
grid on;
set(gca,'FontWeight','bold','LineWidth',2)

%% Taiji Dataset
clc;
clear all;
m = 2;

featureA = 1;
featureB = 2;

[train_features,train_labels,test_features,test_labels] = loadDataset('taiji');


[Y_train, WeightVector_train] = fisherLDA(train_features, train_labels,m);
[Y_test, WeightVector_test] = fisherLDA(test_features, test_labels,m);

N_total = length(test_labels);
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

% Linear Discriminant using decision theory
covar = 0;
for i = 1 : numGroups
    classIndices{i} = find(grp2idx(test_labels) == i);
    N(i) = length(classIndices{i});
    prior(i) = N(i)/N_total;
    mu{i} = mean(X(classIndices{i},:));
    for n = classIndices{i}(1) : classIndices{i}(end)
        covar = covar + (1/(N_total - numGroups)) * (X(classIndices{i},:) - mu{i})' * (X(classIndices{i},:) - mu{i});
    end
end
hold on;

x = [min(X(:,featureA)),max(X(:,featureA))];

for i = 1 : numGroups
    for j = 1 : numGroups
        if i ~= j
            [FisherLd.Coeffs(i,j).slope,FisherLd.Coeffs(i,j).intercept] = fisherlinearDisc(prior(i),prior(j),mu{i},mu{j},covar);
            y = FisherLd.Coeffs(i,j).intercept+ FisherLd.Coeffs(i,j).slope * x;
            plot(x,y, 'LineWidth',2,'DisplayName',sprintf('Class Sep b/w %s,%s',category_names{i},category_names{j}));
            title('{\bf Linear Discriminant Classification}')

        end
    end
end
axis(lim_info)
hold off
grid on;
set(gca,'FontWeight','bold','LineWidth',2)