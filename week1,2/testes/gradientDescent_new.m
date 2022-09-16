function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %size(X)
    theta = theta - (1/m)*alpha*X'*(X*theta - y); % vectorized GD;
    %theta = theta - alpha*X'*(X*theta - y); % vectorized GD
   
##    if (iter >= 2) && ((J_history(iter) - J_history(iter-1)) < 10^(-9))
##      fprintf('\n>> J(theta) < epsilon = 10^-3:  iter = %d\n\n', iter)
##      break
##    end

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

##% initialize J vals to a matrix of 0's
##theta0_vals = -10:0.01:10; theta1_vals = -1:0.01:4;
##J_vals = zeros(length(theta0_vals), length(theta1_vals));
##% Fill out J vals
##for i = 1:length(theta0_vals)
##  for j = 1:length(theta1_vals)
##    t = [theta0_vals(i); theta1_vals(j)];
##    J_vals(i,j) = computeCost(X, y, t);
##    end
##
##end

end
  