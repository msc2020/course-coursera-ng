function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% input layer
% add a_0^(1)
a1 = [ones(m, 1) X]; % obs: X = input, a^(1) = X

% hidden layer
z2 = a1*Theta1'; 
a2 = sigmoid(z2); % sigmoid vectorized

% add a_0^(2)
m2 = size(a2, 1);
a2 = [ones(m2, 1) a2];

% output layer
z3 = a2*Theta2';
a3 = sigmoid(z3);

[val, ind] = max(a3, [], 2); % multiple outputs
p = ind; % vector of predictions from 1 to num_labels

%h = X*all_theta'; % hipostesis vetorized
%A = sigmoid(h); % sigmoid vectorized
%[val, ind] = max(A, [], 2); % multiple outputs
%p = ind; % vector of predictions from 1 to num_labels


% =========================================================================


end
