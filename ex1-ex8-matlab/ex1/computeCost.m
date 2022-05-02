function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


  J = 1/(2*m) * sum((X*theta - y).^2)

  % J l� 1x1 = sum(97x2 * 2x1 - 97 x 1)
  % theta_1 = 1x1 = sum(theta(1) - alpha * 1/m * sum((X*theta - y)) .* X(:, 1))


% =========================================================================

end
