% This file enables to compute the qualtitative error metrics for the point clouds created
% by the output of 'image_to_pcd.py'. It computes the error between ground truth point clouds from lidar and
% RadarHD generated point clouds. The result is shown as a CDF akin to the result in Fig. 5 in the paper

clear;

root_folder = './processed_imgs_13_1_20220320-034822_test_imgs';

trajs = dir(root_folder);
epoch = '120';
chamfer_all_distance = cell(length(trajs),1);
hausdorff_all_distance = cell(length(trajs),1);
mod_hausdorff_all_distance = cell(length(trajs),1);

bin_size = 0;

% Choose the index of the trajectories in 'trajs' you want to test
which_traj = 3:length(trajs);

a = [];
c = [];

for k=1:length(which_traj)
    i = which_traj(k);

    disp(trajs(i).name)
        
    pred_folder = strcat(trajs(i).folder,'/',trajs(i).name,'/',epoch,'/pred/pcd/');    
    pred_file_names = reorder_dir(pred_folder);
    
    label_folder = strcat(trajs(i).folder,'/',trajs(i).name,'/',epoch,'/label/pcd/');
    label_file_names = reorder_dir(label_folder);
    
    chamfer_dist = zeros(length(label_file_names),1);
    mod_hausdorff_dist = zeros(length(label_file_names),1);
    
    for j = 1:length(label_file_names)
        label = pcread(strcat(label_file_names(j).folder,'/',label_file_names(j).name)).Location;
        pred = pcread(strcat(pred_file_names(j).folder,'/',pred_file_names(j).name)).Location;
        chamfer_dist(j) = pc_distance(label(:,1:2),pred(:,1:2),"chamfer",bin_size);
        mod_hausdorff_dist(j) = pc_distance(label(:,1:2),pred(:,1:2),"mod_hausdorff",bin_size);
    end
    chamfer_all_distance{i} = chamfer_dist;
    mod_hausdorff_all_distance{i} = mod_hausdorff_dist;

    a = vertcat(a, chamfer_all_distance{i});
    c = vertcat(c, mod_hausdorff_all_distance{i});
end

figure;
h = cdfplot(a);
set(h,'LineWidth',2,'Color','red')
hold on;
h = cdfplot(c);
set(h,'LineWidth',2,'LineStyle','- -','Color','red')

legend('Chamfer (Ours against Lidar)', 'Mod Hausdorff (Ours against Lidar)')
xlabel('Point Cloud Error (in meters)')
ylabel('CDF')

function file_names = reorder_dir(folder)
    file_names = dir(folder);
    file_names = file_names(3:end,:);
    idx = zeros(length(file_names),1);
    for j=1:length(file_names)
        filename = file_names(j).name;
        pos = find(filename == '_');
        idx(j) = str2num(filename(pos(2)+1:pos(3)-1));        
    end
    [~,pos] = sort(idx);
    file_names = file_names(pos);
end

