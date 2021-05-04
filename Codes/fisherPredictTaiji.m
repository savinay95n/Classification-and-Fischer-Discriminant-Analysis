%% fisherPredictTaiji()
%**************************************************************************
% -------------------------------------------------------------------------
% This function uses Probabilistic Generative Model (Decision Theory), in
% particular, Maximum a Posteriori Estimation to calculate the
% discriminant threshold between classes. This function uses 1 vs all
% scheme od classification. Input belongs to a class which gives the
% maximum threshold. 
% The function is specific to Taiji dataset. The mean of the posterior
% probabilities have been removed since the projected data was biased.
% -------------------------------------------------------------------------
% Inputs: Projected data matrix of Taiji
%         label Vector of Taiji
%         Prior Vector 
%         Class Mean Vector
%         Covariance Matrix
% Outputs: label prediction vector
%          Confusion Matrix
%
% Written by: Savinay Nagendra (sxn265@psu.edu)
%**************************************************************************
%% function
function[prediction,confmat] = fisherPredictTaiji(ProjectedData,labelVector,prior_vector,mu_vector,covar_matrix)

numGroups = length(unique(labelVector));
for i = 1 : numGroups
    for n = 1 : size(ProjectedData,1)
        delta(n,i) = log(prior_vector(i)) - 0.5*(mu_vector{i}/covar_matrix)*mu_vector{i}' + (ProjectedData(n,:)/covar_matrix)*mu_vector{i}';
    end
end
delta = delta - mean(delta);
for i = 1 : numGroups
    for n = 1 : size(ProjectedData,1)
        [~,argmax(n)] = max(delta(n,:));
    end
end

prediction = argmax';
confmat = confusionmat((grp2idx(labelVector)),prediction);
