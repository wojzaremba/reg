clear all
load generated_mats/monochromatic_error.mat
num_colors1 = num_colors;
load generated_mats/layer1_speedup_vs_numcolors_gpu.mat 

baseline = 0.176025;
err = [];
for  nc = 1 : length(num_colors1)
    if sum(num_colors1(nc) == num_colors) > 0
       val = monochromatic_test_error(nc, 1); 
       err(end+1) = val(1);
    end
end

error_increase = err - baseline;

figure1 = figure('Position', [0, 0, 700, 500]);

set(gca,'Fontsize',16);

errorbar(error_increase * 100, mean_speedup, std_speedup, 'bx', 'linewidth', 2);
grid on;
xlabel('Percent loss in performance', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on GPU', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('MattNet first layer approximation: \nEmpirical GPU speedup vs. performance loss'), 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

for  nc = 1 : length(num_colors)
    if num_colors(nc) == 8
        offset = 0.51;
    else
        offset = -0.04;
    end
    text(double(error_increase(nc) * 100 - offset), double(mean_speedup(nc)), sprintf('# colors = %d', num_colors(nc)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end

set(gcf, 'Color', 'w');
export_fig 'paper/img/layer1_GPUspeedup_vs_performance_loss' -pdf
saveas(figure1, 'paper/img/layer1_GPUspeedup_vs_performance_loss', 'epsc');