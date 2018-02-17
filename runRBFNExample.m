% ======== runRBFNExample ========
% This script trains an RBF Network on an example dataset, and plots the
% resulting score function and decision boundary.
% 
% There are three main steps to the training process:
%   1. Prototype selection through k-means clustering.
%   2. Calculation of beta coefficient (which controls the width of the 
%      RBF neuron activation function) for each RBF neuron.
%   3. Training of output weights for each category using gradient descent.
%
% Once the RBFN has been trained this script performs the following:5
%   1. Generates a contour plot showing the output of the category 1 output
%      node.
%   2. Shows the original dataset with the placements of the protoypes and
%      an approximation of the decision boundary between the two classes.
%   3. Evaluates the RBFN's accuracy on the training set.

% $Author: ChrisMcCormick $    $Date: 2014/08/18 22:00:00 $    $Revision: 1.3 $

% Clear all existing variables from the workspace.
clear;

% Remove any holds on the existing plots.
figure(1);
hold off;

figure(2);
hold off;

% Add the subdirectories to the path.
addpath('kMeans');
addpath('RBFN');

% Load the data set. 
% This loads two variables, X and y.
%   X - The dataset, 1 sample per row.
%   y - The corresponding label (category 1 or 2).
% The data is randomly sorted and grouped by category.
data = load('dataset.csv');
X = data(:, 1:4);
y = data(:, 5);

% Set 'm' to the number of data points.
m = size(X, 1);

% ===================================
%     Train RBF Network
% ===================================

disp('Training he RBFN...');

% Train the RBFN using 10 centers per category.
[Centers, betas, Theta] = trainRBFN(X, y, 9, true);
 

% ========================================
%       Measure Training Accuracy
% ========================================

disp('Measuring training accuracy...');

numRight = 0;

wrong = [];

% For each training sample...
for (i = 1 : m)
    % Compute the scores for both categories.
    scores = evaluateRBFN(Centers, betas, Theta, X(i, :))
    
	[maxScore, category] = max(scores)
	
    % Validate the result.
    if (category == y(i))
        numRight = numRight + 1;
    else
        wrong = [wrong; X(i, :)];
    end
    
end

% Mark the incorrectly recognized samples with a black asterisk.
%plot(wrong(:, 1), wrong(:, 2), 'k*');

accuracy = numRight / m * 100;
fprintf('Training accuracy: %d / %d, %.1f%%\n', numRight, m, accuracy);
if exist('OCTAVE_VERSION') fflush(stdout); end;
