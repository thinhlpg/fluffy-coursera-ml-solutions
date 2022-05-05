function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

  #X: 5000x400, need to ad 1
  a1 = [ones(m, 1) X];
  a2 = sigmoid(a1 * Theta1') %5000x400, need to add 1
  a2 = [ones(size(a2), 1) a2]
  h = sigmoid(a2 * Theta2') % 5000x10
  %[probability, predict] = max(h,[],2) %5000x1
  % J = mean Cost for 5000 prediction

  y_mask = zeros(m, num_labels) % 5000x10
  for i = 1:m
    y_mask(i, y(i)) = 1
  end

  J = -1/m * sum(sum(y_mask.*log(h) + (1-y_mask).*log(1-h)))
  %J = -1/m * sum(y_mask'*log(h) + (1-y_mask)'*log(1-h)) why???
  %J = -1/m * ((y==predict)'*log(probability) + (1-(y==predict))'*log(1-probability)) %why??
  J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%

% backward propagatiom for every sample

  for t = 1:m
      % Forward to calculate error for sample t
      a_1 = X(t,:)';
      a_1 = [1; a_1];

      z_2 = Theta1 * a_1;

      a_2 = sigmoid(z_2);
      a_2 = [1; a_2];

      z_3 = Theta2 * a_2;

      a_3 = sigmoid(z_3);

      % Error
      yt = y_mask(t,:)';
      delta_3 = a_3 - yt;
      delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z_2]);
      delta_2 = delta_2(2:end);

      Theta2_grad = Theta2_grad + delta_3 * a_2'
      Theta1_grad = Theta1_grad + delta_2 * a_1
  end


Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad ;

% Add regularization terms
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
