close all;
clear;
clc;

%% Load dataset
dataset = importdata('ESL.arff');
x = dataset.data(:, 1:4);
y = dataset.data(:, 5);

%% Result
tic;
[regresor, theta] = backwardElimination(x, y);
toc;

fprintf('Selected Regressors: \n');
disp(regresor)
fprintf('Parameters: \n');
disp(theta);

%% Forward selestion function method
function [best_reg, best_theta, best_y_proper] = backwardElimination(x, y)
    n = size(x, 2); 
    best_reg = 1:n; 
    best_theta = []; 
    best_y_proper = []; 
    
    while numel(best_reg) > 0
        X_selected = x(:, best_reg);
        theta = pinv(X_selected) * y;
        y_fit = X_selected * theta;
        rss = sum((y - y_fit).^2);
        
        if numel(best_reg) == n
            best_rss = rss;
            best_y_proper = y_fit;
        elseif rss < best_rss
            best_rss = rss;
            best_y_proper = y_fit;
        else
            break; 
        end
        
        best_theta = [best_theta; theta'];
        best_reg_remove = [];
        
        for i = 1:numel(best_reg)
            temp_best_reg = best_reg(best_reg ~= best_reg(i));
            temp_X_selected = x(:, temp_best_reg);
            temp_theta = pinv(temp_X_selected) * y;
            temp_y_fit = temp_X_selected * temp_theta;
            temp_rss = sum((y - temp_y_fit).^2);

            if temp_rss < best_rss
                best_rss = temp_rss;
                best_reg_remove = best_reg(i);
            end
        end

        if ~isempty(best_reg_remove)
            best_reg = best_reg(best_reg ~= best_reg_remove);
        else 
            break; 
        end
    end
    
end
