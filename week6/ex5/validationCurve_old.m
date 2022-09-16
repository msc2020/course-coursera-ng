function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%
m = size(X,1);
J_val_mat = zeros(m, length(lambda_vec));

% normalize the features of the training set
[X_norm, mu, sigma] = featureNormalize(X);

m = length(X_norm(:,1));
n = length(X_norm(1,:));
%m,n

% tranform NaN to zero
for i=1:m
  for j=1:n
    if isnan(X_norm(i,j))==1
      X_norm(i,j)=0;
    endif
  endfor
endfor

%X_norm
%kbhit()

%len_X = size(X); len_X
%len_y = size(y); len_y
%len_Xval = size(Xval); len_Xval
%len_yval = size(yval); len_yval

%error_cv = zeros(length(lambda_vec), 1);
%    
%m,n
%
%p=8;
%%[X_poly] = polyFeatures(X_norm(:,2),p);
%[X_poly] = polyFeatures(X(:,2),p);
%X_poly = [X(:,1) X_poly]; X_poly
%
%[X_poly,mu,sigma] = featureNormalize(X_poly); 
%% tranform NaN to zero
%for i=1:m
%  for j=1:p
%    if isnan(X_poly(i,j)) == 1
%      X_poly(i,j)=0;
%    endif
%  endfor
%endfor
%X_poly
%
%kbhit()

%% compute errors
%for i=1:length(lambda_vec)
%  lambda = lambda_vec(i);
%  %error_train_aux = 10^3; deg_pol = 10^3;
%  %for j=2:n
%  j=8;
%    X_in = X_poly(:,1:j); % select degree of polynomial; obs: degree = j-1
%    X_in
%    [theta_train] = trainLinearReg(X_in, y, lambda);
%    [J_train, grad_train] = linearRegCostFunction(X_in, y, theta_train, 0);
%%    if J_train < error_train_aux
%%      error_train_aux = J_train;
%%      deg_pol = j; i, j, deg_pol, error_train_aux      
%      kbhit()
%%    endif
%  %endfor
%  %error_train(i) = error_train_aux;
%  error_train(i) = J_train;
%%  deg_pol
%%  X_min = X_poly(:,1:deg_pol);
%
%  [theta_val] = trainLinearReg(X, y, lambda);
%  [J_val, grad_val] = linearRegCostFunction(Xval, yval, theta_val, 0);
%  error_val(i) = J_val;    
%  %[error_train, error_val] = learningCurve(X, y, Xval, yval, lambda);
%  %error_train, error_val
%  %endfor    
%endfor
%
%    [theta] = trainLinearReg(X_i, y_i, lambda);
%    [J_val, grad_val] = linearRegCostFunction(Xval, yval, theta, 0);
%    error_val (i) = J_val;

% compute errors
for i=1:length(lambda_vec)
  lambda = lambda_vec(i);
  [theta_train] = trainLinearReg(X, y, lambda);
  [J_train, grad_train] = linearRegCostFunction(X, y, theta_train, 0);
  error_train(i) = J_train;

  [theta_val] = trainLinearReg(X, y, lambda);
  [J_val, grad_val] = linearRegCostFunction(Xval, yval, theta_val, 0);
  error_val(i) = J_val;    
  %[error_train, error_val] = learningCurve(X, y, Xval, yval, lambda);
  error_train, error_val
  
  kbhit()
endfor



error_train, error_val
%error_cv

%error_cv

% obtain min index (i,j) for J_val_mat
%[val_i_min ind_i_min] = min(error_cv);
%[val_min, ind_j_min] = min(val_i_min);
%ind_i_min = ind_i_min(ind_j_min);
%
%ind_i_min, ind_j_min
%
%% obtain error_train and error_val for J_min(lambda_min, theta_min)   
%lambda_min = lambda_vec(ind_j_min);
%for i=1:m
%%  i, m
%  X_min = X(1:m,:); y_min = y(m);
%  [theta_min] = trainLinearReg(X_min, y_min, lambda_min);
%  
%  [J_train_min, grad_train_min] = linearRegCostFunction(X_min, y_min, ...
%  theta_min, 0);
%  error_train(i) = J_train_min;
%  
%  [J_val_min, grad_val_min] = linearRegCostFunction(Xval, yval, theta_min, 0);
%  error_val(i) = J_val_min;    
%endfor
  
  
% =========================================================================

end
