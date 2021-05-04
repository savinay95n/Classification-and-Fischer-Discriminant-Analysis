function [ query_label ] = knnClassifier_anand(train_feature, train_label, query_feature, k)
%Function to implement knn classifier
    % looping for all queries
    n_query = size(query_feature,1);
    query_label = zeros(n_query,1);
    n_train = size(train_feature,1);
    for i=1:n_query
       dist = sum((repmat(query_feature(i,:),n_train,1) - train_feature).^2,2);
       [sort_val, sort_ind] = sort(dist, 'ascend');
       query_label(i) = mode(train_label(sort_ind(1:k)));
    end
end

