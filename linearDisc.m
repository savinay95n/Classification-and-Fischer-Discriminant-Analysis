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