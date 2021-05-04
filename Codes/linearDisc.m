%% linearDisc()
%**************************************************************************
% -------------------------------------------------------------------------
% This function is used to calculate the slope and intercept of the linear
% discriminant between 2 classes for the method of least squares.
% For 2 classes i and j, the discriminant function is calculated using:
% x2 = ((wi0 - wj0) + (wi1 - wj1)*x1)/(wj2 - wi2),
% where the general form of the weight matrix W for 2 features is given by:
%
% W = [wi0 wj0 wk0;  
%      wi1 wj1 wk1; 
%      wi2 wj2 wk2]
%--------------------------------------------------------------------------
% Inputs: WeightMatrix for any two features | [3 x 3]
%         Classi
%         Classj
% Outputs: Slope 
%          Intercept

% Written by: Savinay Nagendra (sxn265@psu.edu)
%**************************************************************************
%% Function
function [slope, intercept] = linearDisc(weightMatrix,class1,class2)
% works only for 2 features
class1_bias = weightMatrix(1,class1);
class2_bias = weightMatrix(1,class2);

threshold = class1_bias - class2_bias;
w1 = weightMatrix(2:3,class1);
w2 = weightMatrix(2:3,class2);

l1 = w1(1) - w2(1);
l2 = w2(2) - w1(2);


slope = l1/l2;
intercept = threshold/l2;


return