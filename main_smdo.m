
% This is the main file for surrogate modeling-based design optimization
% (main_smdo.m)
% Author: Mehnuma Tabassum

% This is a concise version of the code for smdo. For example,
% the experimentation with the cross-validation folds are not shown here,
% just the final folds used for model building. Further, all the data and
% parameter values were changed.


% Part-1: Data Pre-processing
% Part-2: Surrogate Modeling 
% Part-3: Design Optimization


%% Part-1: Data Pre-processing

Total_data = readmatrix("Synthetic Data.csv");  % Reading simulation data
[len, ~] = size(Total_data);                    % Total data points

% Data Scaling
Total_data(:, 1) = (Total_data(:, 1) - min(Total_data(:, 1)))./(max(Total_data(:, 1)) - min(Total_data(:, 1)));
Total_data(:, 2) = (Total_data(:, 2) - min(Total_data(:, 2)))./(max(Total_data(:, 2)) - min(Total_data(:, 2)));
Total_data(:, 3) = (Total_data(:, 3) - min(Total_data(:, 3)))./(max(Total_data(:, 3)) - min(Total_data(:, 3)));
Total_data(:, 4) = (Total_data(:, 4) - min(Total_data(:, 4)))./(max(Total_data(:, 4)) - min(Total_data(:, 4)));
Total_data(:, 5) = (Total_data(:, 5) - min(Total_data(:, 5)))./(max(Total_data(:, 5)) - min(Total_data(:, 5)));
Total_data(:, 6) = (Total_data(:, 6) - min(Total_data(:, 6)))./(max(Total_data(:, 6)) - min(Total_data(:, 6)));
Total_data(:, 7) = (Total_data(:, 7) - min(Total_data(:, 7)))./(max(Total_data(:, 7)) - min(Total_data(:, 7)));
Total_data(:, 8) = (Total_data(:, 8) - min(Total_data(:, 8)))./(max(Total_data(:, 8)) - min(Total_data(:, 8)));
Total_data(:, 9) = (Total_data(:, 9) - min(Total_data(:, 9)))./(max(Total_data(:, 9)) - min(Total_data(:, 9)));
Total_data(:, 10) = (Total_data(:, 10) - min(Total_data(:, 10)))./(max(Total_data(:, 10)) - min(Total_data(:, 10)));



%% Part-2: Surrogate Modeling
% Random Splitting
[num_sample, ~] = size(Total_data);

% Data Randomization
rng default
ind = randperm(num_sample);
ind = randperm(length(ind));
ind = randperm(length(ind));
New_data = Total_data(ind, :);

% 5-fold CV
eq = num_sample/5;
fold1 = New_data(1:eq, :);
fold2 = New_data(eq+1:2*eq, :);
fold3 = New_data(2*eq+1:3*eq, :);
fold4 = New_data(3*eq+1:4*eq, :);
fold5 = New_data(4*eq+1:end, :);

% Train1 = [fold1; fold2; fold3; fold4];
Train2 = [fold2; fold3; fold4; fold5];
% Train3 = [fold3; fold4; fold5; fold1];
Train4 = [fold4; fold5; fold1; fold2];
% Train5 = [fold5; fold1; fold2; fold3];


% Model Building
global gpr_np_fd gpr_np_vg gpr_mass gpr_fatigue

% Model-1: Normal Pressure in Fire Deck
gpr_np_fd = fitrgp(Train2(:, 1:6), Train2(:, 7), 'KernelFunction', 'exponential');

pred1 = predict(gpr_np_fd, fold4(:, 1:6));          
error1 = sqrt(mean((pred1 -fold4(:, 7)).^2));
disp(gpr_np_fd.LogLikelihood)
disp(error1)


% Model-2: 1: Normal Pressure in Valve Guides
gpr_np_vg = fitrgp(Train4(:, 1:6), Train4(:, 8), 'KernelFunction', 'exponential');

pred2 = predict(gpr_np_vg, fold2(:, 1:6));          
error2 = sqrt(mean((pred2 -fold2(:, 8)).^2));
disp(gpr_np_vg.LogLikelihood)
disp(error2)


% Model-3: 1: Design Mass
gpr_mass = fitrgp(Train4(:, 1:6), Train4(:, 9), 'KernelFunction', 'exponential');

pred3 = predict(gpr_mass, fold2(:, 1:6));          
error3 = sqrt(mean((pred3 -fold2(:, 9)).^2));
disp(gpr_mass.LogLikelihood)
disp(error3)


% Model-4: Fatigue Life (%)
gpr_fatigue = fitrgp(Train4(:, 1:6), Train4(:, 10), 'KernelFunction', 'exponential');

pred4 = predict(gpr_fatigue, fold2(:, 1:6));          
error4 = sqrt(mean((pred4 -fold2(:, 10)).^2));
disp(gpr_fatigue.LogLikelihood)
disp(error4)


%% Part-3: Design Optimization

lb = zeros(1, 6);     % Lower bound of design variables
ub = ones(1, 6);      % Upper bound of design variables
nvars = 6;            % Number of design variables
options = optimoptions("gamultiobj","PlotFcn","gaplotpareto", 'PopulationSize', 250);
rng(1, 'twister')     % for reproducibility
[soln, fval, exitflag] = gamultiobj(@objective_func_moo, nvars, [], [], [], [], lb, ub, @cons_moo, options);

