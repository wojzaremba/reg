function load_imagenet_model(type, batch_size)    
    global plan root_path
    if (~exist('type', 'var'))
        type = 'matthew';
    end        
    if (~exist('batch_size', 'var'))
        batch_size = 128;
    end            
    if (exist('plan', 'var') ~= 1) || (isempty(plan)) || (length(plan.layer) < 10)
        json = ParseJSON(sprintf('plans/imagenet_%s.txt', type));
        json{1}.batch_size = batch_size;
		if is_cluster()	
        	Plan(json, sprintf('/misc/vlgscratch3/FergusGroup/denton/imagenet_data/imagenet_%s', type), 1);
		else
        	Plan(json, sprintf('~/imagenet_data/imagenet_%s', type), 0);
        end
        plan.input.step = 1;
    end
end
