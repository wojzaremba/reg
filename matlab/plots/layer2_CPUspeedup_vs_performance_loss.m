clear all
baseline_err = 0.177604;

% Load the baseline time
load generated_mats/layer2_CPU_baseline.mat

% Load the CPU times
load generated_mats/layer2_bisubspace_svd_2_2_CPUspeedups.mat
rank_codes = rank_codes([1:3, 5, 7:10]);
rank_codes_times = rank_codes;
fp_results_speed = fp_results_speed([1:3, 5, 7:10]);

% Load the errors
load generated_mats/layer2_bisubspace_svd_2_2_error.mat
rank_codes_error = rank_codes([1, 3, 5:10]);
errors = errors([1, 3, 5:10]);

mean_speedup = [];
std_speedup = [];
error_increase = [];
codes = {};
for i = 1 : length(rank_codes_times)
   c = rank_codes_times{i};
   
   % Speedup
   speedup = conv2_cpu_time ./ fp_results_speed{i};
   mean_speedup(i) = mean(speedup(2:end));
   std_speedup(i) = std(speedup(2:end));
   
   % Error
   idx = find(ismember(rank_codes_error, c));
   error_increase(i) = errors(idx) - baseline_err;
end
                
figure1 = figure('Position', [0, 0, 700, 600]); hold on;
set(gca,'Fontsize',16);
grid on;
xlabel('Percent loss in performance', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on CPU', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('MattNet first layer approximation: \nEmpirical CPU speedup vs. performance loss'), 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

errorbar(100*error_increase, mean_speedup, std_speedup, 'bx', 'linewidth', 2); 

for i = 1 : length(rank_codes_times)
   code = rank_codes_times{i};
   offset = 0.1;
   text(double(error_increase(i) * 100) + offset,  double(mean_speedup(i)), sprintf('K1 = %s\nK2 = %s', code(3:4), code(9:10)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
end