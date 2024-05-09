close all;
clear;
clc;

%% Load dataset
dataset = importdata('ESL.arff');
x = dataset.data(:, 1:4);
y = dataset.data(:, 5);

%% Result
[regresor, theta] = forwardSelection(x, y);


fprintf('Selected Regressors: \n');
disp(regresor)
fprintf('Parameters: \n');
disp(theta);


%% Forward selestion function method
function [best_reg, best_theta, best_y_proper] = forwardSelection(x, y)
    n = size(x, 2);
    best_rss = Inf; 
    best_reg = []; 
    best_theta = []; 
    best_y_proper = []; 
    
    function [rss, theta, y_proper] = subSelection(selected_reg)
        X_selected = x(:, selected_reg);
        theta = pinv(X_selected) * y;
        y_proper = X_selected * theta;
        rss = sum((y - y_proper).^2);
    end

    function forwardSelect(selected_reg)
        for i = 1:n
            if ~ismember(i, selected_reg)
                new_reg = [selected_reg, i];
                [rss, theta, y_proper] = subSelection(new_reg);
                if rss < best_rss
                    best_rss = rss;
                    best_reg = new_reg;
                    best_theta = theta;
                    best_y_proper = y_proper;
                end
                forwardSelect(new_reg);
            end
        end
    end

    forwardSelect([]);
end


