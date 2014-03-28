function ForwardPass()
global plan
for i = 2:length(plan.layer)
    plan.layer{i}.cpu.vars.X = plan.layer{i - 1}.cpu.vars.out;              
	fptic = tic;
	plan.layer{i}.FP();
    lapse = toc(fptic);
	plan.time.fp(plan.input.step - 1, i) = lapse;
end
