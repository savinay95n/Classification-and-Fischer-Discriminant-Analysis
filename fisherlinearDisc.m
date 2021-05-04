function[slope,intercept] = fisherlinearDisc(priorA,priorB,muA_vector,muB_vector,covar_matrix)

C = log(priorA/priorB) + 0.5*((muB_vector/(covar_matrix))*muB_vector' - (muA_vector/(covar_matrix))*muA_vector');

a = (covar_matrix)\muB_vector' - (covar_matrix)\muA_vector';

slope = -a(1)./a(2);
intercept = C/a(2);

return
