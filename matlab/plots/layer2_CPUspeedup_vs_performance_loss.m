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
                
hplot = errorbar(100*error_increase, mean_speedup, std_speedup, 'b.', 'linewidth', 2); 
set(hplot,'MarkerSize',30); 

for i = 1 : length(rank_codes_times)
   code = rank_codes_times{i};
   if strcmp(code, 'in28_out64')
       offset = -0.9;
   else
     offset = 0.1;
   end
   if strcmp(code, 'in28_out64')
       continue;
   end
   text(double(error_increase(i) * 100) + offset,  double(mean_speedup(i)), sprintf('K_1 = %s\nK_2 = %s', code(3:4), code(9:10)), 'FontSize', 12, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
end


% Data covariance
% Load the baseline time
load generated_mats/layer2_CPU_baseline.mat

% Load the CPU times
load generated_mats/layer2_bisubspace_svd_2_2_CPUspeedups.mat

rank_codes_times = rank_codes([1, 3, 6, 7, 8, 9]);
fp_results_speed = fp_results_speed([1, 3, 6, 7, 8, 9]);

% Load the errors
load generated_mats/layer2_bisubspace_svd_2_2_error_datacov.mat
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
                
hplot = errorbar(100*error_increase, mean_speedup, std_speedup, 'k.', 'linewidth', 2); 
set(hplot,'MarkerSize',30); 

for i = 1 : length(rank_codes_times)
   code = rank_codes_times{i};
   if strcmp(code, 'in28_out64') || strcmp(code, 'in28_out76')
       offset = -0.85;
   else
     offset = 0.1;
   end
   text(double(error_increase(i) * 100) + offset,  double(mean_speedup(i)), sprintf('K_1 = %s\nK_2 = %s', code(3:4), code(9:10)), 'FontSize', 12, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
end

% Finetuned
% Load the baseline time
load generated_mats/layer2_CPU_baseline.mat

% Load the CPU times
load generated_mats/layer2_bisubspace_svd_2_2_CPUspeedups.mat

rank_codes_times = rank_codes([3, 8, 9]);
fp_results_speed = fp_results_speed([3, 8, 9]);

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
                
hplot = errorbar(100*error_increase, mean_speedup, std_speedup, 'r.', 'linewidth', 2); 
set(hplot,'MarkerSize',30); 

for i = 1 : length(rank_codes_times)
   code = rank_codes_times{i};
   offset_vert = 0;
   if strcmp(code, 'in28_out64')
       offset = -0.76;
   elseif strcmp(code, 'in16_out51') || strcmp(code, 'in19_out51')
       offset = -0.82;
   elseif strcmp(code, 'in19_out64')
       offset = -0.8;
       offset_vert = 0.14;
   else
     offset = 0.1;
   end
    
   
   text(double(error_increase(i) * 100) + offset,  double(mean_speedup(i)) + offset_vert, sprintf('K_1 = %s\nK_2 = %s', code(3:4), code(9:10)), 'FontSize', 12, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
end

set(gca,'Fontsize',16);
grid on;
axis([-1, 7, 1.4, 3])
xlabel('Percent loss in performance', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on CPU', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('Second layer approximation: \nEmpirical CPU speedup vs. performance loss'), 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

legend1 = legend('Original', '\Sigma_{data} distance metric', 'Finetuned');
set(legend1, 'Position',[0.552857142857143 0.145166666666668 0.321428571428571 0.128333333333333], 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');


set(gcf, 'Color', 'w');
export_fig '../paper/img/layer2_CPUspeedup_vs_performance_loss_finetune_and_orig' -pdf
saveas(figure1, '../paper/img/layer2_CPUspeedup_vs_performance_loss_finetune_and_orig', 'epsc');

