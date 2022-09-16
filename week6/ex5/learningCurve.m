function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------



%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end


%len_Xval = size(Xval); len_Xval
%len_yval = size(yval); len_yval
%len_X = size(X); len_X
%len_y = size(y); len_y

% compute error train
for i=1:m
%  i, m
  X_i = X(1:i,:); y_i = y(1:i);
  %len_X_i = size(X); len_y_i = size(y_i); len_X_i, len_y_i
  [theta] = trainLinearReg(X_i, y_i, lambda);
  theta_train = theta;
  
  len_theta_train = size(theta_train); %len_theta_train
  [J_train, grad_train] = linearRegCostFunction(X_i, y_i, theta_train, 0);
  error_train(i) = J_train;
  
  theta_cv = theta;
  [J_val, grad_val] = linearRegCostFunction(Xval, yval, theta_cv, 0);
  error_val (i) = J_val;
  % 0/1 misclassification error 
  %h = X_i*theta_train;
  %J_train    
%  error_aux = zeros(i,1);
%  for j=1:i
%    if (h >= 0.5 && y_i == 0) || (h < 0.5 && y_i == 1)
%      error_aux(i,1) = 1;
%      display(">> erro")
%    else
%      error_aux(i,1) = 0;
%      display(">> sem erro")
%    endif
%  endfor
%  error_train(i,1) = (1/m)*sum(error_aux);
  
endfor

%format short e
%error_train
%error_val
%format short


% -------------------------------------------------------------

% =========================================================================

end
