%% binaryCodedTarget()
% *************************************************************************
%--------------------------------------------------------------------------
% This function is used to convert a given label vector into binary 1 of K 
% coded Target Matrix. 
% This function is generic and can work on any given label vector, which 
% might be categorical, numerical or strings.
% Some data sets might have a label vector of strings, belonging to 
% categorical data type, where the strings have not been arranged
% alphabetically (eg. Wallpaper). For convenience of visualization of results, the
% categorical data types are converted to numerical arrays and sorted in
% ascending order. The corresponding feature matrix is also sorted using
% the sorted indices of the label vector. 
% -------------------------------------------------------------------------
% inputs : label vector:of type string, numerical or categorical|[N x K]
% outputs: Binary 1 of K coded Target Matrix | [K x K]
%          Sorted feature matrix | [N x D]

% Written by: Savinay Nagendra (sxn265@psu.edu)
% *************************************************************************
%% Function
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