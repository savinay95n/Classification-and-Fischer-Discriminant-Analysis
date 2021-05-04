% This is the Main program to test and train a Least Squares Classifier
%% Wine dataset

clc;
clear all;
close all;

% Loading Wine Dataset
[wine_train_features, wine_train_labels, wine_test_features, wine_test_labels] = loadDataset('wine');

% Training and Testing using the entire data set  
[weight_matrix_wine, train_pred_wine, test_pred_wine, train_class_mat_wine, test_class_mat_wine,train_conf_mat_wine,test_conf_mat_wine] = LeastSquareClassifierFunction(wine_train_features, wine_train_labels, wine_test_features, wine_test_labels);

% mean group accuracy and std
train_acc = mean(diag(train_class_mat_wine))
train_std = std(diag(train_class_mat_wine));
test_acc = mean(diag(test_class_mat_wine))
test_std = std(diag(test_class_mat_wine));

% training using 2 features for convenience of visualization

% Please enter the features you want to train and test on
featureA = 1;
featureB = 7;

feature_index = [featureA,featureB];
% data set extraction corresponding to the two features
train_featureVector = wine_train_features(:,feature_index);
test_featureVector = wine_test_features(:,feature_index);

% Training and Testing using the two features
[w,tr_pred,tst_pred,tr_class_mat,tst_class_mat,tr_conf_mat,tst_conf_mat] = LeastSquareClassifierFunction(train_featureVector, wine_train_labels, test_featureVector, wine_test_labels);

% mean group accuracy and std
tr_acc = mean(diag(tr_class_mat));
tr_std = std(diag(tr_class_mat));
tst_acc = mean(diag(tst_class_mat));
tst_std = std(diag(tst_class_mat));

category_names = categories(wine_test_labels);
% number of categories in a label vector
numGroups = length(category_names);
colors = jet(numGroups*10);
colors = colors(round(linspace(1,numGroups*10,numGroups)),:);
%Plotting the data points
h1 = gscatter(wine_test_features(:,featureA),wine_test_features(:,featureB),wine_test_labels,'','+o*v^');
for i = 1:numGroups
    h1(i).LineWidth = 2;
    h1(i).MarkerEdgeColor = min(colors(i,:)*1.2,1);
end
lim_info =  [min(wine_test_features(:,featureA)),max(wine_test_features(:,featureA)),...
    min(wine_test_features(:,featureB)),max(wine_test_features(:,featureB))  ];


hold on

x =  [min(wine_test_features(:,featureA)),max(wine_test_features(:,featureA))];

% Linear Discriminants for the two features (one vs one scheme)
for i = 1 : numGroups
    for j = 1 : numGroups
        if i ~= j
            [Ld.Coeffs(i,j).slope,Ld.Coeffs(i,j).intercept] = linearDisc(w,i,j);
            y = Ld.Coeffs(i,j).intercept+ Ld.Coeffs(i,j).slope * x;
            plot(x,y, 'LineWidth',2,'DisplayName',sprintf('Class Sep b/w %s,%s',category_names{i},category_names{j}));
            title('{\bf Linear Discriminant Classification}')
            xlabel('Feature A');
            ylabel('Feature B');
        end
    end
end
axis(lim_info)
hold off
grid on;
set(gca,'FontWeight','bold','LineWidth',2)
%% Wallpaper dataset

clc;
clear all;
close all;
% Loading Wallpaper Dataset
[wall_train_features, wall_train_labels, wall_test_features, wall_test_labels] = loadDataset('wallpaper');

% Training and Testing using the entire data set  
[weight_matrix_wall, train_pred_wall, test_pred_wall, train_class_mat_wall, test_class_mat_wall,train_conf_mat_wall, test_conf_mat_wall] = LeastSquareClassifierFunction(wall_train_features, wall_train_labels, wall_test_features, wall_test_labels);

% mean group accuracy and std
train_acc = mean(diag(train_class_mat_wall))
train_std = std(diag(train_class_mat_wall));
test_acc = mean(diag(test_class_mat_wall))
test_std = std(diag(test_class_mat_wall));

% training for 2 features for convenience of visualization

% Please enter the features you want to train and test on
featureA = 1;
featureB = 7;

feature_index = [featureA,featureB];
% data set extraction corresponding to the two features
train_featureVector = wall_train_features(:,feature_index);
test_featureVector = wall_test_features(:,feature_index);

% Training and Testing using the two features
[w,tr_pred,tst_pred,tr_class_mat,tst_class_mat,tr_conf_mat,tst_conf_mat] = LeastSquareClassifierFunction(train_featureVector, wall_train_labels, test_featureVector, wall_test_labels);

% mean group accuracy and std
tr_acc = mean(max(tr_class_mat));
tr_std = std(max(tr_class_mat));
tst_acc = mean(max(tst_class_mat));
tst_std = std(max(tst_class_mat));

category_names = categories(wall_test_labels);
% number of categories in a label vector
numGroups = length(category_names);
colors = jet(numGroups*10);
colors = colors(round(linspace(1,numGroups*10,numGroups)),:);
%Plotting the data points
h1 = gscatter(wall_test_features(:,featureA),wall_test_features(:,featureB),wall_test_labels,'','+o*v^');
for i = 1:numGroups
    h1(i).LineWidth = 2;
    h1(i).MarkerEdgeColor = min(colors(i,:)*1.2,1);
end
lim_info =  [min(wall_test_features(:,featureA)),max(wall_test_features(:,featureA)),...
    min(wall_test_features(:,featureB)),max(wall_test_features(:,featureB))  ];


hold on

x =  [min(wall_test_features(:,featureA)),max(wall_test_features(:,featureA)) ];

% Linear Discriminants for the two features (one vs one scheme)
for i = 1 : numGroups
    for j = 1 : numGroups
        if i ~= j
            [Ld.Coeffs(i,j).slope,Ld.Coeffs(i,j).intercept] = linearDisc(w,i,j);
            y = Ld.Coeffs(i,j).intercept+ Ld.Coeffs(i,j).slope * x;
            plot(x,y, 'LineWidth',2,'DisplayName',sprintf('Class Sep b/w %s,%s',category_names{i},category_names{j}));
            title('{\bf Linear Discriminant Classification}')
            xlabel('Feature A');
            ylabel('Feature B');
        end
    end
end
axis(lim_info)
hold off
grid on;
set(gca,'FontWeight','bold','LineWidth',2)


%% Taiji dataset

clc;
clear all;
close all;

% Loading Taiji Dataset
[taiji_train_features, taiji_train_labels, taiji_test_features, taiji_test_labels] = loadDataset('taiji');

% Training and Testing using the entire data set  
[weight_matrix_taiji, train_pred_taiji, test_pred_taiji, train_class_mat_taiji, test_class_mat_taiji,train_conf_mat_taiji, test_conf_mat_taiji] = LeastSquareClassifierFunction(taiji_train_features, taiji_train_labels, taiji_test_features, taiji_test_labels);

% mean group accuracy and std
test_acc = mean(diag(test_class_mat_taiji))
test_std = std(diag(test_class_mat_taiji));
train_acc = mean(diag(train_class_mat_taiji))
train_std = std(diag(train_class_mat_taiji));

% training for 2 features for convenience of visualization

% Please enter the features you want to train and test on
featureA = 1;
featureB = 7;

feature_index = [featureA,featureB];
% data set extraction corresponding to the two features
train_featureVector = taiji_train_features(:,feature_index);
test_featureVector = taiji_test_features(:,feature_index);

% Training and Testing using the two features
[w,tr_pred,tst_pred,tr_class_mat,tst_class_mat,tr_conf_mat,tst_conf_mat] = LeastSquareClassifierFunction(train_featureVector, taiji_train_labels, test_featureVector, taiji_test_labels);

% mean group accuracy and std
tr_acc = mean(diag(tr_class_mat));
tr_std = std(diag(tr_class_mat));
tst_acc = mean(diag(tst_class_mat));
tst_std = std(diag(tst_class_mat));

category_names = categories(taiji_test_labels);
% number of categories in a label vector
numGroups = length(category_names);
colors = jet(numGroups*10);
colors = colors(round(linspace(1,numGroups*10,numGroups)),:);
%Plotting the data points
h1 = gscatter(taiji_test_features(:,featureA),taiji_test_features(:,featureB),taiji_test_labels,'','+o*v^');
for i = 1:numGroups
    h1(i).LineWidth = 2;
    h1(i).MarkerEdgeColor = min(colors(i,:)*1.2,1);
end
lim_info =  [min(taiji_test_features(:,featureA)),max(taiji_test_features(:,featureA)),...
    min(taiji_test_features(:,featureB)),max(taiji_test_features(:,featureB))  ];


hold on

x =  [min(taiji_test_features(:,featureA)),max(taiji_test_features(:,featureA)) ];

% Linear Discriminants for the two features (one vs one scheme)
for i = 1 : numGroups
    for j = 1 : numGroups
        if i ~= j
            [Ld.Coeffs(i,j).slope,Ld.Coeffs(i,j).intercept] = linearDisc(w,i,j);
            y = Ld.Coeffs(i,j).intercept+ Ld.Coeffs(i,j).slope * x;
            plot(x,y, 'LineWidth',2,'DisplayName',sprintf('Class Sep b/w %s,%s',category_names{i},category_names{j}));
            title('{\bf Linear Discriminant Classification}')
            xlabel('Feature A');
            ylabel('Feature B');
        end
    end
end
axis(lim_info)
hold off
grid on;
set(gca,'FontWeight','bold','LineWidth',2)

