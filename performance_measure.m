function [Accuracy, TP,FP] = performance_measure(pred_y, y)
% Function calculates the overall accuracy, true
%positive and false positive counts based on the predicted and actual labels

TP = length(find(y == 1 & pred_y == 1)); %True positive
FP = length(find(y == 0 & pred_y == 1)); %False positive
errors = abs(y - pred_y);
err = sum(errors);
Accuracy = (1 - err / length(y))*100;%Accuracy

end

