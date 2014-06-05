clear all
figure1 = figure('Position', [0, 0, 700, 600]);
set(gca,'Fontsize',16);
set(gca, 'ytick', 1:0.2:2.6)

hold on;

baseline = 0.177604;
colors_to_plot = [6, 12, 16, 24];
load generated_mats/monochromatic_error.mat
num_colors_err = num_colors_list;
load generated_mats/layer1_speedup_vs_numcolors_gpu.mat 
num_colors_time = num_colors;

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
        offset = -0.5;
    else
        offset = + 0.2;
    end
    if colors_to_plot(nc) == 6 || colors_to_plot(nc) == 16
        offset_vert = -0.02;
    else 
        offset_vert = 0;
    end
    text(double(error_inc(nc) * 100 + offset), double(mean_speedup_to_plot(nc)) + offset_vert, sprintf('%d', colors_to_plot(nc)), 'FontSize', 14, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end


% With data covariance
hold on
baseline = 0.177604;

colors_to_plot = [3, 4, 6, 12, 16, 24];
load generated_mats/monochromatic_error_datacov.mat
num_colors_err = num_colors_list;
load generated_mats/layer1_speedup_vs_numcolors_gpu.mat 
num_colors_time = num_colors;

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
    if colors_to_plot(nc) == 16
        offset = -0.5;
    else
        offset = 0.2;
    end
    if colors_to_plot(nc) == 6 || colors_to_plot(nc) == 16
        offset_vert = -0.02;
    else 
        offset_vert = 0;
    end
    text(double(error_inc(nc) * 100 + offset), double(mean_speedup_to_plot(nc)) + offset_vert, sprintf('%d', colors_to_plot(nc)), 'FontSize', 14, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end

% With finetuning
hold on
baseline = 0.177604;


baseline = 0.177604;
colors_to_plot = [4, 6, 12];
load generated_mats/monochromatic_finetuned_error.mat
num_colors_err = num_colors_list;
load generated_mats/layer1_speedup_vs_numcolors_gpu.mat 
num_colors_time = num_colors;

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
        offset = -0.5;
    else
        offset = 0.2;
    end
    if colors_to_plot(nc) == 6
        offset_vert = -0.02;
    else 
        offset_vert = 0;
    end
    text(double(error_inc(nc) * 100 + offset), double(mean_speedup_to_plot(nc)) + offset_vert, sprintf('%d', colors_to_plot(nc)), 'FontSize', 14, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end

grid on;
axis([-1, 7, 1, 2.6]);
xlabel('Percent loss in performance', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on GPU', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('First layer approximation: \nEmpirical GPU speedup vs. performance loss'), 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

legend1 = legend('Original', '\Sigma_{data} distance metric', 'Finetuned');
set(legend1, 'Position',[0.555714285714286 0.138500000000002 0.321428571428571 0.128333333333333], 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

set(gcf, 'Color', 'w');
export_fig '../paper/img/layer1_GPUspeedup_vs_performance_loss_finetune_and_orig' -pdf
saveas(figure1, '../paper/img/layer1_GPUspeedup_vs_performance_loss_finetune_and_orig', 'epsc');