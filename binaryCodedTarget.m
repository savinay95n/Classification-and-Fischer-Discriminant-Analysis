function[Target,sorted_feature_matrix] = binaryCodedTarget(label_vector,feature_matrix)

[sorted_label_vector,Index] = sort(label_vector);
sorted_feature_matrix = feature_matrix(Index,:);
[categories,category_starting_indices,~] = unique(sorted_label_vector);
Target = zeros(length(label_vector),length(categories));
i = 1;
while(i~=length(categories))
    Target(category_starting_indices(i):category_starting_indices(i+1)-1,i) = 1;
    i = i + 1;
end
Target(category_starting_indices(i):length(label_vector),end) = 1;

return