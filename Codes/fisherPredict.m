%% fisherPredict()
%**************************************************************************
% -------------------------------------------------------------------------
% This function uses Probabilistic Generative Model (Decision Theory), in
% particular, Maximum a Posteriori Estimation to calculate the
% discriminant threshold between classes. This function uses 1 vs all
% scheme od classification. Input belongs to a class which gives the
% maximum threshold.
% -------------------------------------------------------------------------
% Inputs: Projected data matrix | [N x D']
%         label Vector [N x 1]
%         Prior Vector [K x 1]
%         Class Mean Vector [D x 1]
%         Covariance Matrix [D x D]
% Outputs: label prediction vector [N x 1]
%          Confusion Matrix [K x K]
%
% Written by: Savinay Nagendra (sxn265@psu.edu)
%**************************************************************************
%% function
function[prediction,confmat] = fisherPredict(ProjectedData,labelVector,prior_vector,mu_vector,covar_matrix)

numGroups = length(unique(labelVector));
for i = 1 : numGroups
    for n = 1 : size(ProjectedData,1)
        delta(n,i) = log(prior_vector(i)) - 0.5*(mu_vector{i}/covar_matrix)*mu_vector{i}' + (ProjectedData(n,:)/covar_matrix)*mu_vector{i}';
        [~,argmax(n)] = max(delta(n,:));

    end
end

prediction = argmax';
confmat = confusionmat((grp2idx(labelVector)),prediction);
