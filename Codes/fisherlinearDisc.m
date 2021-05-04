%% fisherlinearDisc()
% *************************************************************************
% -------------------------------------------------------------------------
% This function is used to plot the linear discriminant between two classes
% using probabilistic enerative model. The slope an dintervcept of the
% discriminant function asre provided as outputs.
% -------------------------------------------------------------------------
% Inputs: priorA | [1 x 1]
%         prior B | [1 x 1]
%         class mean vector A | [D x 1]
%         class mean vector B | [D x 1]
%         covariance matrix | [D x D]  
%Outputs: slope
%         Intercept
%
% Written by: Savinay Nagendra (sxn265@psu.edu)
%**************************************************************************
%% function
function[slope,intercept] = fisherlinearDisc(priorA,priorB,muA_vector,muB_vector,covar_matrix)

C = log(priorA/priorB) + 0.5*((muB_vector/(covar_matrix))*muB_vector' - (muA_vector/(covar_matrix))*muA_vector');

a = (covar_matrix)\muB_vector' - (covar_matrix)\muA_vector';

slope = -a(1)./a(2);
intercept = C/a(2);

return
