// This file helps in visualizing thresholded, cartesian format images as point cloud
// It contrasts the quality of RadarHD output and ground truth label point clouds

close all;
clear;

root_folder = './processed_imgs_13_1_20220320-034822_test_imgs/';

trajs = dir(root_folder);
epoch = '120';

figure;
set(gcf,'position',[0,0,1000,1000])

for i=7:length(trajs)
    pred_folder = strcat(trajs(i).folder,'/',trajs(i).name,'/',epoch,'/pred/pcd/');    
    pred_file_names = reorder_dir(pred_folder);
    
    label_folder = strcat(trajs(i).folder,'/',trajs(i).name,'/',epoch,'/label/pcd/');
    label_file_names = reorder_dir(label_folder);

    for j = 1:length(label_file_names)
        label = pcread(strcat(label_file_names(j).folder,'/',label_file_names(j).name)).Location;
        pred = pcread(strcat(pred_file_names(j).folder,'/',pred_file_names(j).name)).Location;

        subplot(1,2,1);
        scatter(label(:,2), label(:,1));
        title(strcat('Traj No ' , trajs(i).name, ' Label'))
        axis equal
        grid on
        xlim([-10.8,10.8]); ylim([0,10.8]);
        
        subplot(1,2,2);
        scatter(pred(:,2), pred(:,1));
        title(strcat('Traj No ' , trajs(i).name, ' RadarHD'))
        axis equal
        grid on
        xlim([-10.8,10.8]); ylim([0,10.8]);
        
        
        pause(0.01);
    end
end

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
