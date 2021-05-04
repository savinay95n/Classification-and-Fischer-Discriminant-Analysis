function[projectedVector,WeightVector] = LDA(feature_matrix,label_vector,m)
category_names = categories(label_vector);
numGroups = length(category_names);
dim = size(feature_matrix,2);
tr_labels_array = grp2idx(label_vector);
total_mean = 0;
temp = 0;


for i = 1 : numGroups
    classIndices{i} = find(tr_labels_array == i);
    N(i) = length(classIndices{i});
    X{i} = feature_matrix(classIndices{i},:);
    pre_proj_mean{i} = mean(X{i});
    total_mean = total_mean + N(i) * pre_proj_mean{i}/length(label_vector);
    temp = temp + N(i) * (pre_proj_mean{i}' - total_mean') * (pre_proj_mean{i}' - total_mean')';
    interClassCovariance = temp;
end
temp = 0;
intraClassCovariance = zeros(dim);
for i = 1:numGroups
    for n = 1:N(i)
        temp = temp + (X{i}(n,:)' - pre_proj_mean{i}')*(X{i}(n,:)' - pre_proj_mean{i}')';
        intra_covariance{i} = temp;
        intraClassCovariance = intraClassCovariance + intra_covariance{i};
    end
end

[WeightVector, LAMBDA] = eig(interClassCovariance,intraClassCovariance,'qz');
lambda = diag(LAMBDA);
[~,sortIndex] = sort(lambda,'descend');
WeightVector = WeightVector(:,sortIndex);
WeightVector = WeightVector(:,1:m);

projectedVector = feature_matrix * WeightVector;


return