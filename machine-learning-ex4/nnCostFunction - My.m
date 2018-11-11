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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y_label = zeros(10,m);

[hiddens,h_theta] = hTheta(X,Theta1,Theta2);

% 将y转成十个神经元的矩阵
for i = 1:m
  y_label(y(i),i) = y(i);
end

% 计算正则项代价函数的正则项
regu_item = lambda * (nn_params' * nn_params) / (2 * m);

% 计算代价函数
J = sum(- log(h_theta(:)) .* y_label(:) - log(1 - h_theta(:)) .* (1 - y_label(:))) / m + regu_item;

% 计算残差
error_output_unroll = h_theta(:) - y_label(:);
error_output = reshape(error_output_unroll,num_labels,m);
error_hidden = Theta2' * error_output;

% 计算delta
delta_theta2 = zeros(num_labels,hidden_layer_size + 1);
delta_theta1 = zeros(hidden_layer_size,input_layer_size + 1);

% 计算梯度
for i =  1:size(error_output,1)
  delta_theta2(i,:) = sum(error_output(i,:)' .* Theta2(i,:));  % 对Theta2的第i行求梯度的delta，一行的权值个数等于隐藏层神经元个数，error_output(i,:)有5000列，对5000列求总的delta，下同
  Theta2_grad(i,:) = delta_theta2(i,:) / m + lambda * [0 Theta2(i,2:end)];
end

for i = i:size(error_hidden,1) - 1 % 在bp求权值的的delta时，右侧神经元（比如这里求input到hidden的weight时，hidden处的神经元）要把偏置神经元的残差errot去除，所以 -1
  delta_theta1(i,:) = sum(error_hidden(i,:)' .* Theta1(i,:)); 
  Theta1_grad(i,:) = delta_theta1(i,:) / m + lambda * [0 Theta1(i,2:end)];
end

##for i = 1:m
##  % 计算代价函数
##%   J = J - (log(h_theta(i,:)) * y_label(i,:)' + log(1 - h_theta(i,:)) * (1 - y_label(i,:))');
##  
##  % 计算残差
##  error_output = h_theta(i,:) - y_label(i,:);
##  error_hidden = error_output * Theta2 .* [ones(1,1) sigmoidGradient(hiddens(i,:))];
##  
##  % 计算梯度
##  delta_theta2 = delta_theta2 + error_output' * [ones(1,1) sigmoidGradient(hiddens(i,:))];
##  delta_theta1 = delta_theta1 + error_hidden(:,2:end)' * [ones(1,1) X(i,:)];
##  Theta2_grad = delta_theta2 / m + lambda * Theta2;
##  Theta1_grad = delta_theta1 / m + lambda * Theta1;
##end

% J = sum(log(hTheta(X)) .* y_label' + log(1 - hTheta(X)) * (1 - y)');

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
