function new_featureVector = fisher_proj(input_featureVector, input_labels, ...
    numClasses)

    total_mean = mean(input_featureVector);
    length_each_class = [countcats(input_labels)];
    temp = 0;
    cols = size(input_featureVector,2);
    S_W = zeros(cols);
    S_k = cell(1,3);
    temp_sk_mat = zeros(1,cols);
    S_B = zeros(cols);
    for k=1:numClasses
        data{k} = input_featureVector(temp+1:temp+length_each_class(k),:);
        class_mean{k} = mean(data{k});
        for j=1:length_each_class(k)   
            temp_sk_mat = temp_sk_mat+((data{k}(j,:)-class_mean{k})'*(data{k}(j,:)-class_mean{k}));
        end
        S_k{k} = temp_sk_mat;
        S_W = S_W + S_k{k};
        S_B = S_B + (length_each_class(k)*((class_mean{k}-total_mean)'*(class_mean{k}-total_mean)));
        temp = temp+length_each_class(k);  
    end

    [eigVec eigVal] = eig(S_B,S_W, 'qz');

    eigVal = diag(eigVal);
    eigmat = [eigVec eigVal];
    sorted_eigmat = sortrows(eigmat,-(cols+1));

    % new_dim = input('Enter the dimension that you want it to be reduced to');
    new_dim = numClasses-1;

    new_W = sorted_eigmat(:,1:new_dim);
    new_featureVector = input_featureVector * new_W;
end