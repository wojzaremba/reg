baseline = 0.177604;
global plan;
load_imagenet_model;

figure1 = figure('Position', [0, 0, 700, 600]); hold on;
hold on;
set(gca,'Fontsize',16);

% Plot FC1
load generated_mats/FC1_approx_errors.mat
rank_list = rank_list(3:2:end);
errors = errors(3:2:end);
[N, M] = size(plan.layer{13}.cpu.vars.W);
orig_weights = prod(N*M) + M; % +M for bias
approx_weights = N * rank_list + rank_list * M + M;
weight_savings = orig_weights ./ approx_weights;
plot(100 * (errors - baseline), weight_savings, 'r', 'linewidth', 2);


% Plot FC2
load generated_mats/FC2_approx_errors.mat
rank_list = rank_list(3:2:end);
errors = errors(3:2:end);
[N, M] = size(plan.layer{14}.cpu.vars.W);
orig_weights = prod(N*M) + M; % +M for bias
approx_weights = N * rank_list + rank_list * M + M;
weight_savings = orig_weights ./ approx_weights;
plot(100 * (errors - baseline), weight_savings, 'b', 'linewidth', 2);

% Plot FC2
load generated_mats/FC3_approx_errors.mat
rank_list = rank_list(3:2:end);
errors = errors(3:2:end);
[N, M] = size(plan.layer{14}.cpu.vars.W);
orig_weights = prod(N*M) + M; % +M for bias
approx_weights = N * rank_list + rank_list * M + M;
weight_savings = orig_weights ./ approx_weights;
plot(100 * (errors - baseline), weight_savings, 'k', 'linewidth', 2);


xlabel('Percent loss in performance', 'FontSize', 18, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Reduction is number of weights', 'FontSize', 18, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('MattNet FC layers approximation: \nPerformance loss vs. reduction of parameters'), 'FontSize', 20, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
legend1 = legend('FC1', 'FC2', 'FC3');