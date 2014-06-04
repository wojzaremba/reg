clear all
baseline_err = 0.177604;
figure1 = figure('Position', [0, 0, 700, 600]); hold on;
hold on;

% Load the baseline time
load generated_mats/layer2_CPU_baseline.mat

% Load the CPU times
load generated_mats/layer2_bisubspace_svd_2_2_CPUspeedups.mat
% Remove 10, 4, 2, 5
rank_codes_times = rank_codes([1, 3, 6:9]);
fp_results_speed = fp_results_speed([1, 3, 6:9]);

% Load the errors
load generated_mats/layer2_bisubspace_svd_2_2_error.mat
% Remove 9, 4, 3, 5
rank_codes_error = rank_codes([1:2, 6:8, 10]);
errors = errors([1:2, 6:8, 10]);

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
                
%errorbar(100*error_increase, mean_speedup, std_speedup, 'bx', 'linewidth', 2); 
plot(100*error_increase, mean_speedup, 'bo', 'linewidth', 3); 

for i = 1 : length(rank_codes_times)
   code = rank_codes_times{i};
   if strcmp(code, 'in28_out64')
       offset = -0.75;
   else
     offset = 0.1;
   end
   text(double(error_increase(i) * 100) + offset,  double(mean_speedup(i)), sprintf('K1 = %s\nK2 = %s', code(3:4), code(9:10)), 'FontSize', 13, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
end

% Finetuned
% Load the baseline time
load generated_mats/layer2_CPU_baseline.mat

% Load the CPU times
load generated_mats/layer2_bisubspace_svd_2_2_CPUspeedups.mat

rank_codes_times = rank_codes([3, 9]);
fp_results_speed = fp_results_speed([3, 9]);

% Load the errors
load generated_mats/layer2_bisubspace_svd_2_2_finetuned_error.mat
rank_codes_error = rank_codes;

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
                
%errorbar(100*error_increase, mean_speedup, std_speedup, 'rx', 'linewidth', 2); 
plot(100*error_increase, mean_speedup, 'rx', 'linewidth', 2);

for i = 1 : length(rank_codes_times)
   code = rank_codes_times{i};
   if strcmp(code, 'in28_out64')
       offset = -0.75;
   else
     offset = 0.1;
   end
   text(double(error_increase(i) * 100) + offset,  double(mean_speedup(i)), sprintf('K1 = %s\nK2 = %s', code(3:4), code(9:10)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
end

set(gca,'Fontsize',16);
grid on;
axis([-0.5, 7, 1.4, 3])
xlabel('Percent loss in performance', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on CPU', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('MattNet first layer approximation: \nEmpirical CPU speedup vs. performance loss'), 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
