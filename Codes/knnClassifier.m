% function accuracy = knnClassifier(train_featureVector,train_labels,test_featureVector,test_labels, k)
function [ query_label ] = knnClassifier(train_feature, train_label, query_feature, k) 
%     N = size(train_featureVector, 1);
%     M = size(test_featureVector,1);
%     count=0;
%     
%     for i=1:M
%         distance = sqrt(sum((train_featureVector - repmat(test_featureVector(i,:), N, 1)).^2, 2));
%         [val,ind] = sort(distance,'ascend');
%         nearest_neigbrs = ind(1:k);
% %             x_closest = train_featureVector(nearest_neigbrs,:);
%         x_closestLabels = train_labels(nearest_neigbrs);
%         pred_label(i) = mode(x_closestLabels);
%         if pred_label(i) == test_labels(i)
%             count=count+1;
%         end
%     end
%    accuracy = pred_label';
    n_query = size(query_feature,1);
    query_label = zeros(n_query,1);
    n_train = size(train_feature,1);
    for i=1:n_query
       dist = sum((repmat(query_feature(i,:),n_train,1) - train_feature).^2,2);
       [sort_val, sort_ind] = sort(dist, 'ascend');
       query_label(i) = mode(train_label(sort_ind(1:k)));
    end   
    
end
