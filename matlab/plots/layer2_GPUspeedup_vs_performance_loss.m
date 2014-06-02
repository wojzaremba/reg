load generated_mats/mattnet_fp_results_maha.mat
load generated_mats/layer2_speedup_vs_rank_gpu.mat

baseline = 0.176025;
err = [];
for rank = 8:16
   val = fp_results_maha(sprintf('48_2_%d', rank)); 
   err(end+1) = val(1);
end

error_increase = err - baseline;

mean_speedup = mean(speedup_vs_rank');
std_speedup = std(speedup_vs_rank');

figure1 = figure('Position', [0, 0, 700, 600]);


errorbar(error_increase * 100, mean_speedup, std_speedup, 'bx', 'linewidth', 2); grid on;
set(gca,'Fontsize',16);
xlabel('Percent loss in performance', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on GPU', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('MattNet second layer approximation: \nPerformance loss vs. empirical GPU speedup'), 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
legend1 = legend('G = 48; H = 2');
set(legend1, 'Position',[0.151428571428572 0.855666666666668 0.215714285714286 0.0433333333333333], 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

ranks = 8:16;
for r = 1 : length(ranks)
    if (ranks(r) < 10)
        offset = 0.25;
    else
        offset= 0.28;
    end
    if (ranks(r) == 15 || ranks(r) == 13)
        offset = -0.06;
    end
   text(double(error_increase(r) * 100 - offset), double(mean_speedup(r)), sprintf('rank = %d', ranks(r)), 'FontSize', 10, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
end
set(gca,'XTick',[0.5:0.5:3.5]);

set(gcf, 'Color', 'w');
export_fig 'paper/img/layer2_GPUspeedup_vs_performance_loss' -pdf
saveas(figure1, 'paper/img/layer2_GPUspeedup_vs_performance_loss', 'epsc');