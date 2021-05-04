function [W] = Fischer_anand(X_train, Y_train)
% Finds Fischer projection for data
% X_train - N x D input features of train data
% Y_train - N x 1 labels of train data    

num_classes = length(unique(Y_train));

% mean vector m_i of each class
m_i = zeros(num_classes, size(X_train,2));
for i=1:num_classes
    % find class i points
    i_idx = find(Y_train == categorical(i));
    x_i = X_train(i_idx,:);
    m_i(i,:) = sum(x_i,1) / size(x_i,1); 
end

% computing SW (total within class covariance)
SW = zeros(size(X_train,2));
for i=1:num_classes
   % find class i points
   i_idx = find(Y_train == categorical(i));
   x_i = X_train(i_idx,:); 
   for j=1:size(x_i,1)
    SW = SW + ((x_i(j,:) - m_i(i,:))' * (x_i(j,:) - m_i(i,:)));
   end
end

% computing SB (between class covariance)
total_mean = sum(X_train,1) / size(X_train,1);
SB = zeros(size(X_train,2));
for i=1:num_classes
    % find class i points
    i_idx = find(Y_train == categorical(i));
    Nk = size(i_idx,1);
    SB = SB + Nk*(m_i(i,:) - total_mean)'*(m_i(i,:) - total_mean);
end
% computing Fischer projection W = inv(SW) SB
[eig_vec, eig_val] = eig(SB, SW, 'qz');
% sort eig_vals and pick 'd' largest eig_val
d = num_classes - 1;
[sort_val, sort_idx] = sort(diag(eig_val),'descend');
W = eig_vec(:,sort_idx(1:d));
end

