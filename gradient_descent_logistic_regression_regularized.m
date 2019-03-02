function [theta, J, h] = gradient_descent_logistic_regression_regularized(x_norm, y, theta, alpha, num_of_iter, lambda )
%%performs the gradient descent operation for a logistic regression with regularization problem for ‘n’ iterations
%% Inputs
% theta :includes both theta_0 and so on
% x_norm: normalized feature vector
% y: (actual value)
% alpha : learning rate
% num_of_iter : Number of iterations
% lamda :(Regularization Parameter)
%%Outpus
%J : cost at every iteration [Array variable]
%theta : updated weights

%%
% computing the length of our actual vlue
m = size(x_norm,1);

% Creating a zero matrix to store cost value
J = zeros(num_of_iter, 1);

% Gradient descent
for i=1:num_of_iter
    
    % Cost at every iteration
    J(i)=compute_cost_logistic_regression_regularized(theta,x_norm,y,lambda);
    
    % Hypothesis
    h=compute_sigmoid(x_norm*theta);
    
    % Gradient Descent (theta=theta-alpha/m*delta)
    delta=(h-y)'*x_norm(:,2:end);
    
    
    % Update theta
    theta(1)=theta(1)-(alpha/m)*sum(h-y);
    theta(2:end)=theta(2:end)*(1-lambda*alpha/m)-(alpha/m)*delta';
    
end

end