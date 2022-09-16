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
    if iter == 1
      fprintf('\niter = %3d:  theta_0 = [%f  %f], alpha = %f\n\n', iter-1, theta(1,1), theta(2,1), alpha)
    end
    
    
    theta = theta - alpha*X'*(X*theta - y); % vectorized GD
    if iter > 1
      fprintf('iter = %3d:  theta = [%f  %f],  J(theta) = %f \n', iter-1, theta(1,1), theta(2,1), J_history(iter-1))
    end
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
