clear all
baseline_speed = [4.2650;
                    4.1625;
                    4.2896;
                    4.3330;
                    4.2603;
                    4.1942;
                    4.3887;
                    4.4028;
                    4.2429;
                    4.1578;
                    4.2091;
                    4.4068;
                    4.3280;
                    4.2958;
                    4.2025;
                    4.1173;
                    4.2602;
                    4.1808;
                    4.2460;
                    4.0924;
                    4.4416;
                    4.3803;
                    4.3206;
                    4.1500;
                    4.2214;
                    4.2566;
                    4.1708;
                    4.3740;
                    4.4177;
                    4.4961;
                    4.4024;
                    4.4150]
                
load generated_mats/mattnet_bisubsvd_fp_results.mat
figure1 = figure('Position', [0, 0, 700, 600]); hold on;
errors = [];
mean_speedup = [];
std_speedup = [];
keep_keys = {};
baseline_err = 0.176025;
keys = fp_results_error.keys();
for k = 1 : length(keys)
   key = keys{k};
   speed = fp_results_speed(key);
   err = fp_results_error(key);
   if length(speed) < 32
       continue
   end
   if key(8) == '4'
       continue
   end
   ms =  mean(baseline_speed(2:end) ./ speed(2:end));
   if err > 0 && err < .23 && ((err < .185 && ms > 1.4) || (err < .2 && ms > 1.5) || (err < 0.22 && ms > 1.5) )%( (err < 0.22 && ms > 1.6) || (err < 0.2 && ms > 1.5) || (err < 0.19 && ms > 1.4)|| (err < 0.185 && ms > 1.3) ) && err > 0
       mean_speedup(end+1) = ms;
       std_speedup(end+1) = std(4 ./ speed(2:end));
       errors(end+1) = err;
       fprintf('%s\n', key);
       keep_keys{end+1} = key;
   end
       
end

error_increase = 100 * (errors - baseline_err);

errorbar(error_increase, mean_speedup, std_speedup, 'bx', 'linewidth', 2); 
text(double(error_increase(1)) + 0.04, double(mean_speedup(1)), sprintf('input rank = %d\noutput rank = %d', floor(96 * 0.4 / 2), floor(256 * 0.5 / 2)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
text(double(error_increase(2)) - 0.55, double(mean_speedup(2)), sprintf('input rank = %d\noutput rank = %d', floor(96 * 0.4 / 2), floor(256 * 0.6 / 2)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
text(double(error_increase(3)) + 0.04, double(mean_speedup(3)), sprintf('input rank = %d\noutput rank = %d', floor(96 * 0.5 / 2), floor(256 * 0.5 / 2)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
text(double(error_increase(4)) -0.55, double(mean_speedup(4)), sprintf('input rank = %d\noutput rank = %d', floor(96 * 0.5 / 2), floor(256 * 0.6 / 2)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
text(double(error_increase(5)) + 0.04, double(mean_speedup(5)), sprintf('input rank = %d\noutput rank = %d', floor(96 * 0.6 / 2), floor(256 * 0.5 / 2)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
text(double(error_increase(6)) + 0.04, double(mean_speedup(6)), sprintf('input rank = %d\noutput rank = %d', floor(96 * 0.6 / 2), floor(256 * 0.6 / 2)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

errors = [];
mean_speedup = [];
std_speedup = [];
keep_keys = {};
baseline_err = 0.176025;
keys = fp_results_error.keys();
for k = 1 : length(keys)
   key = keys{k};
   speed = fp_results_speed(key);
   err = fp_results_error(key);
   if length(speed) < 32
       continue
   end
   if key(8) == '2'
       continue
   end
   ms =  mean(baseline_speed(2:end) ./ speed(2:end));
   if err > 0 && err < .23 && ((err < .2 && ms > 1.4) || (err < 0.25 && ms > 1.5) )%( (err < 0.22 && ms > 1.6) || (err < 0.2 && ms > 1.5) || (err < 0.19 && ms > 1.4)|| (err < 0.185 && ms > 1.3) ) && err > 0
       mean_speedup(end+1) = ms;
       std_speedup(end+1) = std(4 ./ speed(2:end));
       errors(end+1) = err;
       fprintf('%s\n', key);
       keep_keys{end+1} = key;
   end
       
end

error_increase = 100 * (errors - baseline_err);

errorbar(error_increase, mean_speedup, std_speedup, 'rx', 'linewidth', 2); 

text(double(error_increase(1)) + 0.04, double(mean_speedup(1)), sprintf('input rank = %d\noutput rank = %d', floor(96 * 0.4 / 2), floor(256 * 0.6 / 2)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
text(double(error_increase(2)) + 0.04, double(mean_speedup(2)), sprintf('input rank = %d\noutput rank = %d', floor(96 * 0.4 / 2), floor(256 * 0.7 / 2)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

grid on;
set(gca,'Fontsize',16);
set(gca,'XTick',[0.5:0.5:3.5]);
set(gca,'YTick',[1.3:0.1:1.9]);
xlabel('Percent loss in performance', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on CPU', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('MattNet second layer approximation: \nPerformance loss vs. empirical CPU speedup'), 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
legend1 = legend('G = 2; H = 2', ...
                 'G = 2; H = 4');
set(legend1, 'Position',[0.151428571428572 0.855666666666668 0.215714285714286 0.0433333333333333], 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
axis([0.5, 3.5, 1.3, 1.9]);
   

% text(double(error_increase(1)) -0.4, double(mean_speedup(1)), sprintf('input rank = %d\noutput rank = %d', floor(96 * 0.4 / 2), floor(256 * 0.5 / 2)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
% text(double(error_increase(2)) + 0.04, double(mean_speedup(2)), sprintf('input rank = %d\noutput rank = %d', floor(96 * 0.4 / 2), floor(256 * 0.6 / 2)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
% text(double(error_increase(3)) + 0.04, double(mean_speedup(3)), sprintf('input rank = %d\noutput rank = %d', floor(96 * 0.5 / 2), floor(256 * 0.6 / 2)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
% text(double(error_increase(4)) + 0.04, double(mean_speedup(4)), sprintf('input rank = %d\noutput rank = %d', floor(96 * 0.6 / 2), floor(256 * 0.6 / 2)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');


set(gcf, 'Color', 'w');
export_fig 'paper/img/layer2_CPUspeedup_vs_performance_loss' -pdf
saveas(figure1, 'paper/img/layer2_CPUspeedup_vs_performance_loss', 'epsc');