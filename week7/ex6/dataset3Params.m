function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% selected values
values_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';

% errors
error_train = zeros(length(values_vec), 1);
error_val = zeros(length(values_vec), 1);

K = length(values_vec);
erro_cv = zeros(K, K);

% compute errors
for i=1:K
  C = values_vec(i);  
  for j=1:K
    sigma = values_vec(j);  
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); %model
    predictions = svmPredict(model, Xval); %predictions
    error_cv(i,j) = mean(double(predictions ~= yval)); %error_cv   
  endfor  
endfor

%error_cv

% obtain min index (i,j) for J_val_mat
[val_i_min ind_i_min] = min(error_cv);
[val_min, ind_j_min] = min(val_i_min);
ind_i_min = ind_i_min(ind_j_min);
%ind_i_min, ind_j_min

C_min = values_vec(ind_i_min); sigma_min = values_vec(ind_j_min);
%C_min, sigma_min

C = C_min; sigma = sigma_min;


% =========================================================================

end