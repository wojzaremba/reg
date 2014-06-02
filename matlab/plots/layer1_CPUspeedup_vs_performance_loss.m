clear all
load generated_mats/monochromatic_error.mat
num_colors1 = [8, 12, 16, 24, 32];
monochromatic_test_error = errors;
load generated_mats/layer1_sppedup_vs_numcolors_cpu.mat 

mean_speedup = mean(monochromatic_cpu_speedup');
std_speedup = std(monochromatic_cpu_speedup');
baseline = 0.177604;

err = [];
mean_speedup_plot = [];
std_speedup_plot = [];

for  nc = 1 : length(num_colors1)
    idx = find(num_colors_list == num_colors1(nc));
    val = errors(idx); 
    err(end+1) = val(1);   
    
    idx = find(num_colors == num_colors1(nc));
    sp = mean_speedup(idx);
    st = std_speedup(idx);
    mean_speedup_plot(end+1) = sp;
    std_speedup_plot(end+1) = st;
end

error_increase = err - baseline;



figure1 = figure('Position', [0, 0, 700, 500]);

set(gca,'Fontsize',16);

errorbar(error_increase * 100, mean_speedup_plot, std_speedup_plot, 'bx', 'linewidth', 2);
grid on;
xlabel('Percent loss in performance', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on CPU', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('MattNet first layer approximation: \nEmpirical CPU speedup vs. performance loss'), 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

for  nc = 1 : length(num_colors1)
    if num_colors1(nc) >= 32 || num_colors1(nc) == 12
        offset = -0.07;
    elseif num_colors1(nc) == 8
        offset = 0.58;
    else
        offset = 0.63;
    end
    text(double(error_increase(nc) * 100 - offset), double(mean_speedup_plot(nc)), sprintf('# colors = %d', num_colors(nc)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end
% 
% set(gcf, 'Color', 'w');
% export_fig 'paper/img/layer1_CPUspeedup_vs_performance_loss' -pdf
% saveas(figure1, 'paper/img/layer1_CPUspeedup_vs_performance_loss', 'epsc');