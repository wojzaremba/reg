function ret = is_cluster()
    if (feature('numCores') <= 4) % There are 4 cores on mac pro.
        fprintf('We are running locally\n');
        ret = false;
    else
        fprintf('We are on a cluster\n');
        ret = true;   
    end
end