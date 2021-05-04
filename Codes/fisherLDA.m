%% fisherLDA()
% *************************************************************************
% -------------------------------------------------------------------------
% This function performs Fisher's Linear Discriminant Analysis on the given
% data and provides Projected Feature Matrix as output.
% -------------------------------------------------------------------------
% Inputs: feature Matrix | [N x D]
%         label vector | [N x 1]
%         projected subspace dimension [1 x 1]
%Outputs: Projected feature matrix [N x D']
%         Weight Vector [D x D']

% Written by: Savinay Nagendra (sxn265@psu.edu)
% *************************************************************************
%% function
function[Y,WeightVector] = fisherLDA(featureMatrix, labelVector, m)
numGroups = length(unique(labelVector));
totalMean = mean(featureMatrix);
temp = 0;
for i = 1 : numGroups
    classIndices{i} = find(grp2idx(labelVector) == i);
    N(i) = length(classIndices{i});
    classMean{i} = mean(featureMatrix(classIndices{i},:));
    temp = temp + N(i) * (classMean{i}' - totalMean') * (classMean{i}' - totalMean')';
    SB = temp;
end
temp = 0;
SW = 0;
for i = 1 : numGroups
    for n = classIndices{i}(1) : classIndices{i}(end)
         temp = temp + (featureMatrix(n,:)' - classMean{i}') * (featureMatrix(n,:)' - classMean{i}')';
         Sk{i} = temp;
    end
    temp = 0;
    SW = SW + Sk{i};
end

[WeightVector, lambdaMatrix]=eig(SB,SW,'qz'); 
lambda=diag(lambdaMatrix);

[~, SortIndex] = sort(lambda,'descend');

WeightVector = WeightVector(:,SortIndex); 
WeightVector = WeightVector(:,1:m); 
Y = featureMatrix*WeightVector; 