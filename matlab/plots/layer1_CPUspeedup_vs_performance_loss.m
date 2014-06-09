clear all

figure1 = figure('Position', [0, 0, 900, 600]);
set(gca,'Fontsize',16);
set(gca, 'ytick', 1.4:0.2:3)
hold on;

baseline = 0.177604;
colors_to_plot = [6, 8, 12, 16, 24];
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


hplot = errorbar(error_inc * 100, mean_speedup_to_plot, std_speedup_to_plot, 'b.', 'linewidth', 2);
set(hplot,'MarkerSize',30); 

for  nc = 1 : length(colors_to_plot)
    if colors_to_plot(nc) == 6
        offset = -0.6;
    else
        offset = 0.1;
    end
    if colors_to_plot(nc) == 24
        offset_vert = -0.03;
    else 
        offset_vert = 0;
    end
    text(double(error_inc(nc) * 100 + offset), double(mean_speedup_to_plot(nc)) + offset_vert, sprintf('C'' = %d', colors_to_plot(nc)), 'FontSize', 14, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end


% With data covariance
hold on
baseline = 0.177604;

colors_to_plot = [3, 4, 6, 8, 12, 16, 24];
load generated_mats/monochromatic_error_datacov.mat
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

hplot = errorbar(error_inc * 100, mean_speedup_to_plot, std_speedup_to_plot, 'k.', 'linewidth', 2);
set(hplot,'MarkerSize',30); 

for  nc = 1 : length(colors_to_plot)
    if colors_to_plot(nc) == 16 || colors_to_plot(nc) == 3 || colors_to_plot(nc) == 4
        offset = -0.6;
    else
        offset = 0.1;
    end
    if colors_to_plot(nc) == 24
        offset_vert = -0.03;
    else 
        offset_vert = 0;
    end
    text(double(error_inc(nc) * 100 + offset), double(mean_speedup_to_plot(nc)) + offset_vert, sprintf('C'' = %d', colors_to_plot(nc)), 'FontSize', 14, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end


% With finetuning
hold on
baseline = 0.177604;


baseline = 0.177604;
colors_to_plot = [4, 6, 8, 12];
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

hplot = errorbar(error_inc * 100, mean_speedup_to_plot, std_speedup_to_plot, 'r.', 'linewidth', 2);
set(hplot,'MarkerSize',30); 

for  nc = 1 : length(colors_to_plot)
    if colors_to_plot(nc) == 12
        offset = -0.6;
    else
        offset = 0.1;
    end
    if colors_to_plot(nc) == 24
        offset_vert = -0.02;
    else 
        offset_vert = 0;
    end
    text(double(error_inc(nc) * 100 + offset), double(mean_speedup_to_plot(nc)) + offset_vert, sprintf('C'' = %d', colors_to_plot(nc)), 'FontSize', 14, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end

grid on;
axis([-1, 6, 1.4, 3]);
xlabel('Percent loss in performance', 'FontSize', 20, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on CPU', 'FontSize', 20, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('First layer approximation: \nPerformance loss vs. empirical CPU speedup'), 'FontSize', 24, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

legend1 = legend('Original', '||W||_{data} distance metric', 'Finetuned');
set(legend1, 'Position',[0.5899999515594 0.146666666666667 0.287777777777778 0.211666666666666], 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

set(gcf, 'Color', 'w');
export_fig '../paper/img/layer1_CPUspeedup_vs_performance_loss_finetune_and_orig' -pdf
saveas(figure1, '../paper/img/layer1_CPUspeedup_vs_performance_loss_finetune_and_orig', 'epsc');