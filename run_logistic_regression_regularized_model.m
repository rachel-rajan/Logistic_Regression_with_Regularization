%run_logistic_regression_regularized_model.m

close all;
clear all;
clc;
dataset = load('Sample Data.txt');

%Feature vector
x = dataset(:,1:2);

%desired value
y = dataset(:,3);

% Find indices of positive and negative examples of true label
y_pos_idx = find(y == 1);
y_neg_idx = find(y == 0);

%Plot the data
plot(x(y_pos_idx, 1), x(y_pos_idx, 2),'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
hold on;
plot(x(y_neg_idx, 1), x(y_neg_idx, 2),'ko', 'MarkerFaceColor', 'g', 'MarkerSize', 7);
xlabel('Feature x1')
ylabel('Feature x2')
title('Distribution of the Data')
legend('Positive class', 'Negative class')

%Normalize features
x_norm = normalize_features(x);

%Regularization parameter
lambda = [0;1;100;1000];

%% 
% Adding column of ones to x matrix

x_norm = [ones(length(x_norm), 1) x_norm];
theta = [0;0;0];
alpha = 0.1;
num_of_iter = 3000;

for j=1:length(lambda)

% Running gradient descent operation for a logistic regression problem for 1000 iterations
[theta, J] = gradient_descent_logistic_regression_regularized(x_norm, y, theta, alpha, num_of_iter, lambda(j));

%Plotting the cost function
figure;
plot(1:num_of_iter, J);
xlabel('Number of iterations')
ylabel('Cost Value')
title(sprintf('Cost Function Plot for Lambda = %d', lambda(j)));

%% Performance measure

%target prediction
pred_y=x_norm*theta; 

% probability calculation
[h]= compute_sigmoid(pred_y); % Hypothesis Function
pred_y(h>=0.5)=1;
pred_y(h<0.5)=0;

[Accuracy, TP,FP] = performance_measure(pred_y, y);
fprintf('Performance Evaluations : %d \n ');
fprintf('For Lambda = %d \n ', lambda(j));
fprintf('Cost value is %d \n ', min(J));
theta
fprintf('Accuracy is %d percentage\n ', Accuracy);
fprintf('True Positive Count : %d \n ', TP);
fprintf('False Positive Count : %d \n ', FP);

end

% Find indices of positive and negative samples in predicted ones
pred_y_pos_idx = find(pred_y == 1);
pred_y_neg_idx = find(pred_y == 0);

%% Plot

figure; 
hold on;
plot(x(y_pos_idx, 1), x(y_pos_idx, 2),'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
plot(x(y_neg_idx, 1), x(y_neg_idx, 2),'ko', 'MarkerFaceColor', 'g', 'MarkerSize', 7);
plot(x(pred_y_pos_idx, 1), x(pred_y_pos_idx, 2),'k*', 'LineWidth', 0.8, 'MarkerSize', 7);
plot(x(pred_y_neg_idx, 1), x(pred_y_neg_idx, 2),'k+', 'LineWidth', 0.8, 'MarkerSize', 7);
legend('Positive Class -True Label', 'Negative Class -True Label', 'Positive Class -Predicted Label', 'Negative Class -Predicted Label');
xlabel('Feature x1')
ylabel('Feature x2')
title('Visualization of Performance')
hold off;

