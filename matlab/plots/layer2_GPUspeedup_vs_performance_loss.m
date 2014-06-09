clear all
figure1 = figure('Position', [0, 0, 900, 600]);
set(gca,'Fontsize',16);
hold on;

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

hplot = errorbar(error_inc * 100, mean_speedup_to_plot, std_speedup_to_plot, 'b.', 'linewidth', 2);
set(hplot,'MarkerSize',30); 

for  r = 1 : length(ranks_to_plot)
    if ranks_to_plot(r) == 6
        offset = -0.5;
    else
        offset = 0.1;
        
    end
    if ranks_to_plot(r) == 16
        offset_vert = -0.05;
    else 
        offset_vert = 0;
    end
    text(double(error_inc(r) * 100 + offset), double(mean_speedup_to_plot(r)) + offset_vert, sprintf('K = %d', ranks_to_plot(r)), 'FontSize', 12, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end

% Mahalanobis
baseline = 0.177604;
ranks_to_plot = [5, 6, 8, 10, 12, 14, 16];
load generated_mats/layer2_bisubspace_48_2_error_maha.mat
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

hplot = errorbar(error_inc * 100, mean_speedup_to_plot, std_speedup_to_plot, 'k.', 'linewidth', 2);
set(hplot,'MarkerSize',30); 

for  r = 1 : length(ranks_to_plot)
    if ranks_to_plot(r) == 6 || ranks_to_plot(r) == 5
        offset = -0.5;
    elseif ranks_to_plot(r) == 10 || ranks_to_plot(r) == 12 || ranks_to_plot(r) == 16 || ranks_to_plot(r) == 14
        offset = -0.6;
    else
        offset = 0.1;
        
    end
    if ranks_to_plot(r) == 16
        offset_vert = -0.05;
    else 
        offset_vert = 0;
    end
    text(double(error_inc(r) * 100 + offset), double(mean_speedup_to_plot(r)) + offset_vert, sprintf('K = %d', ranks_to_plot(r)), 'FontSize', 12, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end


% Finetuned

baseline = 0.177604;
ranks_to_plot = [5, 6, 8];
load generated_mats/layer2_bisubspace_finetuned_error.mat
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

hplot = errorbar(error_inc * 100, mean_speedup_to_plot, std_speedup_to_plot, 'r.', 'linewidth', 2);
set(hplot,'MarkerSize',30); 

for  r = 1 : length(ranks_to_plot)
    if ranks_to_plot(r) == 6
        offset = -0.5;
    else
        offset = 0.1;
        
    end
    offset_vert = 0;
    text(double(error_inc(r) * 100 + offset), double(mean_speedup_to_plot(r)) + offset_vert, sprintf('K = %d', ranks_to_plot(r)), 'FontSize', 12, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

end

grid on;
axis([-0.8, 7, 1, 2.8]);
set(gca,'Fontsize',16);
xlabel('Percent loss in performance', 'FontSize', 20, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Empirical gain in speed on GPU', 'FontSize', 20, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
title(sprintf('Second layer approximation: \nPerformance loss vs. empirical GPU speedup'), 'FontSize', 24, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');


legend1 = legend('Original', '||W||_{maha} distance metric', 'Fine-tuned');
set(legend1, 'Position',[0.581111111111111 0.15 0.297482095691881 0.220000000000002], 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

set(gcf, 'Color', 'w');
export_fig '../paper/img/layer2_GPUspeedup_vs_performance_loss_finetune_and_orig' -pdf
saveas(figure1, '../paper/img/layer2_GPUspeedup_vs_performance_loss_finetune_and_orig', 'epsc');

