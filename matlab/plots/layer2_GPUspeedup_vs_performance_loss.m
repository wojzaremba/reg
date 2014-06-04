clear all

baseline = 0.177604;
ranks_to_plot = [6, 8, 10, 12, 14, 16];
load generated_mats/layer2_bisubspace_48_2_error.mat
rank_list_err = rank_list;
load generated_mats/layer2_speedup_vs_rank_gpu.mat 
rank_list_time = rank_list;

error_inc = [];
mean_speedup_to_plot = [];
std_speedup_to_plot = [];
for  nc = 1 : length(ranks_to_plot)
    idx = find(rank_list_err == ranks_to_plot(nc));
    if isempty(idx)
        continue
    else
        error_inc(nc) = errors(idx);
    end
    idx = find(rank_list_time == ranks_to_plot(nc));
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

set(gca,'Fontsize',16);
xlabel('Percent loss in performance', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on GPU', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('MattNet second layer approximation: \nPerformance loss vs. empirical GPU speedup'), 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
legend1 = legend('G = 48; H = 2');
set(legend1, 'Position',[0.151428571428572 0.855666666666668 0.215714285714286 0.0433333333333333], 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

for  r = 1 : length(ranks_to_plot)
    if ranks_to_plot(r) == 6
        offset = -1;
    else
        offset = 0.2;
        
    end
    offset_vert = 0;
    text(double(error_inc(r) * 100 + offset), double(mean_speedup_to_plot(r)) + offset_vert, sprintf('K = %d', ranks_to_plot(r)), 'FontSize', 14, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end
set(gcf, 'Color', 'w');
export_fig '../paper/img/layer2_GPUspeedup_vs_performance_loss_finetune_and_orig' -pdf
saveas(figure1, '../paper/img/layer2_GPUspeedup_vs_performance_loss_finetune_and_orig', 'epsc');