clear all

baseline = 0.177604;
colors_to_plot = [6, 12, 16, 24];
load generated_mats/monochromatic_error.mat
num_colors_err = num_colors_list;
load generated_mats/layer1_speedup_vs_numcolors_cpu.mat 
num_colors_time = num_colors_list;

error_inc = [];
mean_speedup_to_plot = [];
std_speedup_to_plot = [];
for  nc = 1 : length(colors_to_plot)
    idx = find(num_colors_err == colors_to_plot(nc));
    if isempty(idx)
        continue
    else
        error_inc(nc) = errors(idx);
    end
    idx = find(num_colors_time == colors_to_plot(nc));
    if isempty(idx)
        continue
    else
        mean_speedup_to_plot(nc) = mean_speedup(idx);
        std_speedup_to_plot(nc) = std_speedup(idx);
    end
end

error_inc = error_inc - baseline;

figure1 = figure('Position', [0, 0, 700, 500]);

set(gca,'Fontsize',16);

errorbar(error_inc * 100, mean_speedup_to_plot, std_speedup_to_plot, 'bx', 'linewidth', 2);
grid on;
xlabel('Percent loss in performance', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on CPU', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('MattNet first layer approximation: \nEmpirical GPU speedup vs. performance loss'), 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

for  nc = 1 : length(colors_to_plot)
    if colors_to_plot(nc) == 6
        offset = -2.2;
    else
        offset = + 0.35;
    end
    if colors_to_plot(nc) == 6 || colors_to_plot(nc) == 16
        offset_vert = -0.02;
    else 
        offset_vert = 0;
    end
    text(double(error_inc(nc) * 100 + offset), double(mean_speedup_to_plot(nc)) + offset_vert, sprintf('# colors = %d', colors_to_plot(nc)), 'FontSize', 14, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end

% set(gcf, 'Color', 'w');
% export_fig '../paper/img/layer1_CPUspeedup_vs_performance_loss_nofinetune' -pdf
% saveas(figure1, '../paper/img/layer1_CPUspeedup_vs_performance_loss_nofinetune', 'epsc');

% With finetuning
hold on
baseline = 0.177604;


baseline = 0.177604;
colors_to_plot = [4, 6, 12];
load generated_mats/monochromatic_finetuned_error.mat
num_colors_err = num_colors_list;
load generated_mats/layer1_speedup_vs_numcolors_cpu.mat 
num_colors_time = num_colors_list;

error_inc = [];
mean_speedup_to_plot = [];
std_speedup_to_plot = [];
for  nc = 1 : length(colors_to_plot)
    idx = find(num_colors_err == colors_to_plot(nc));
    if isempty(idx)
        continue
    else
        error_inc(nc) = errors(idx);
    end
    idx = find(num_colors_time == colors_to_plot(nc));
    if isempty(idx)
        continue
    else
        mean_speedup_to_plot(nc) = mean_speedup(idx);
        std_speedup_to_plot(nc) = std_speedup(idx);
    end
end

error_inc = error_inc - baseline;
% 
% figure1 = figure('Position', [0, 0, 700, 500]);
% 
% set(gca,'Fontsize',16);

errorbar(error_inc * 100, mean_speedup_to_plot, std_speedup_to_plot, 'rx', 'linewidth', 2);
grid on;
xlabel('Percent loss in performance', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on GPU', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('MattNet first layer approximation: \nEmpirical CPU speedup vs. performance loss'), 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

for  nc = 1 : length(colors_to_plot)
    if colors_to_plot(nc) == 12
        offset = -2;
    else
        offset = 0.2;
    end
    if colors_to_plot(nc) == 6
        offset_vert = -0.02;
    else 
        offset_vert = 0;
    end
    text(double(error_inc(nc) * 100 + offset), double(mean_speedup_to_plot(nc)) + offset_vert, sprintf('# colors = %d', colors_to_plot(nc)), 'FontSize', 13, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end

legend1 = legend('Original error', 'Finetuned error');
set(legend1, 'Position',[0.653571428571429 0.792833333333335 0.24 0.097], 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

set(gcf, 'Color', 'w');
export_fig '../paper/img/layer1_CPUspeedup_vs_performance_loss_finetune_and_orig' -pdf
saveas(figure1, '../paper/img/layer1_CPUspeedup_vs_performance_loss_finetune_and_orig', 'epsc');