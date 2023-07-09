// This is a helper file to compute various point cloud comparison metrics

function distance = pc_distance(pc_A, pc_B, type, bin_size)
    pc_A = bin_pc(pc_A, bin_size);
    pc_B = bin_pc(pc_B, bin_size);
    
    if type == "chamfer"
        distanceA = 0;
        for i=1:size(pc_A,1)
            distanceA = distanceA + min(sqrt(sum((pc_B-pc_A(i,:)).^2,2)));
        end
        distanceA = distanceA / size(pc_A,1);
        
        distanceB = 0;
        for i=1:size(pc_B,1)
            distanceB = distanceB + min(sqrt(sum((pc_A-pc_B(i,:)).^2,2)));
        end
        distanceB = distanceB / size(pc_B,1);
        
        distance = 0.5*distanceA + 0.5*distanceB;
        
    elseif type == "hausdorff"
        distanceA = zeros(size(pc_A,1),1);
        for i=1:size(pc_A,1)
            distanceA(i) = min(sqrt(sum((pc_B-pc_A(i,:)).^2,2)));
        end

        distanceB = zeros(size(pc_B,1),1);
        for i=1:size(pc_B,1)
            distanceB(i) = min(sqrt(sum((pc_A-pc_B(i,:)).^2,2)));
        end
        distance = max([max(distanceA), max(distanceB)]);

    elseif type == "mod_hausdorff"
        distanceA = zeros(size(pc_A,1),1);
        for i=1:size(pc_A,1)
            distanceA(i) = min(sqrt(sum((pc_B-pc_A(i,:)).^2,2)));
        end

        distanceB = zeros(size(pc_B,1),1);
        for i=1:size(pc_B,1)
            distanceB(i) = min(sqrt(sum((pc_A-pc_B(i,:)).^2,2)));
        end
        distance = max([median(distanceA), median(distanceB)]);
    elseif type == "l1"
        RMAX = 10.8;
        x_axis_new_grid = 0:bin_size:RMAX;
        y_axis_new_grid = -RMAX:bin_size:RMAX;        
        map_A = zeros(length(x_axis_new_grid),length(y_axis_new_grid));
        map_B = zeros(length(x_axis_new_grid),length(y_axis_new_grid));
        for i=1:size(pc_A,1)
            x_loc = find(x_axis_new_grid >= pc_A(i,1),1);
            y_loc = find(y_axis_new_grid >= pc_A(i,2),1);
            map_A(x_loc,y_loc) = 1;
        end
        for i=1:size(pc_B,1)
            x_loc = find(x_axis_new_grid >= pc_B(i,1),1);
            y_loc = find(y_axis_new_grid >= pc_B(i,2),1);
            map_B(x_loc,y_loc) = 1;
        end
        distance = sum(sum(abs(map_A-map_B))); 
    end
end

function new_pc = bin_pc(pc, bin_size)
    if bin_size == 0
        new_pc = pc;
    else
        RMAX = 10.8;
        new_pc = zeros(size(pc,1),3);
        x_axis_new_grid = 0:bin_size:RMAX;
        y_axis_new_grid = -RMAX:bin_size:RMAX;
        for i=1:size(new_pc,1)
            new_pc(i,1) = x_axis_new_grid(find(x_axis_new_grid>=pc(i,1),1));
            new_pc(i,2) = y_axis_new_grid(find(y_axis_new_grid>=pc(i,2),1));
        end
    end
end