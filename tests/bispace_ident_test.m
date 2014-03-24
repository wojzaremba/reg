global root_path plan debug
clearvars -except root_path plan W B
debug = 2;
init();
load_imagenet_model(); % By default, loads Matt's model.


if (~exist('W', 'var'))
    W = plan.layer{5}.cpu.vars.W;
    B = plan.layer{5}.cpu.vars.B;
end

iclust = 4;
iratio = 1;
oclust = 2;
oratio = 1;

json = ParseJSON('plans/imagenet_matthew.txt');
json{5}.type = 'BiclusteredSVDConv';
json{5}.iclust = iclust;
json{5}.iratio = iratio;
json{5}.oclust = oclust;
json{5}.oratio = oratio;

Plan(json, '~/imagenet_data/imagenet_matthew', 0);

perm_in = 1:size(W, 4);
perm_out = 1:size(W, 1);

oclust_sz = size(W, 1) / oclust;
iclust_sz = size(W, 4) / iclust;

odegree = floor(size(W, 1) * oratio / oclust);
idegree = floor(size(W, 4) * iratio / iclust);

C = zeros(size(W, 4) / iclust, idegree, iclust, oclust);
Z = zeros(odegree, size(W, 2), size(W, 3), idegree, iclust, oclust);
F = zeros(size(W, 1) / oclust, odegree, iclust, oclust);

for i = 1 : iclust
    for o = 1 : oclust
        Z(:, :, :, :, i, o) = W(((o - 1) * oclust_sz + 1) : (o * oclust_sz), : , :, ((i - 1) * iclust_sz + 1) : (i * iclust_sz));
        C(:, :, i, o) = eye(size(C, 1));
        F(:, :, i, o) = eye(size(F, 1));
    end
end

plan.layer{5}.cpu.vars.perm_in = single(perm_in);
plan.layer{5}.cpu.vars.perm_out = single(perm_out);
plan.layer{5}.cpu.vars.B = single(B);
plan.layer{5}.cpu.vars.C = single(C);
plan.layer{5}.cpu.vars.Z = single(Z);
plan.layer{5}.cpu.vars.F = single(F);

% Get error
plan.time.fp = 0;
error = 0;
plan.input.step = 1;

plan.input.GetImage(0);
ForwardPass(); 
error = error + plan.classifier.GetScore(5);
fprintf('%d / %d = %f \n', error, i * plan.input.batch_size, error / (i * plan.input.batch_size));
plan.time.fp