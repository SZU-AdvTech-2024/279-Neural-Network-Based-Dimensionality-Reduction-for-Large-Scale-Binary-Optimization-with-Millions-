function [c_indx,c_point] = ExactKmeans(feature,K)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Xiangyu Wang (email: xiangyu.wang@uni-bielefeld.de)

    % ClusteringS
    N1 = size(feature,1);
    % Finding the max and min value
    [f_max] = max(feature);
    [f_min] = min(feature);
    feature(:,1) = (feature(:,1)-f_min(1))/(f_max(1) - f_min(1));
    feature(:,2) = (feature(:,2)-f_min(2))/(f_max(2) - f_min(2));
    [f_max] = max(feature);
    [f_min] = min(feature);
    A1 = [f_max(1),f_max(2)];
    B1 = [f_max(1),f_min(2)];
    C1 = [f_min(1),f_max(2)];
    D1 = [f_min(1),f_min(2)];
    A  = A1;
    B  = B1;
    C  = C1;
    D  = D1;
    for i = 1 : 100
        dis(:,1) = sqrt((feature(:,1)-A(1)).^2+(feature(:,2)-A(2)).^2);
        dis(:,2) = sqrt((feature(:,1)-B(1)).^2+(feature(:,2)-B(2)).^2);
        dis(:,3) = sqrt((feature(:,1)-C(1)).^2+(feature(:,2)-C(2)).^2);
        dis(:,4) = sqrt((feature(:,1)-D(1)).^2+(feature(:,2)-D(2)).^2);
        for j = 1:N1
            [~,indx(j)] = min(dis(j,:));
        end
        A = mean(feature(indx==1,:),1);
        B = mean(feature(indx==2,:),1);
        C = mean(feature(indx==3,:),1);
        D = mean(feature(indx==4,:),1);
        A(isnan(A)) = A1(isnan(A));
        B(isnan(B)) = B1(isnan(B));
        C(isnan(C)) = C1(isnan(C));
        D(isnan(D)) = D1(isnan(D));
    end
    c_indx  = indx;
    c_point = [A;B;C;D];

    figure;
    hold on;
    colors = ['r', 'g', 'b', 'm']; % 定义 4 种颜色
    markers = ['o', 's', 'd', '^']; % 不同簇使用不同的标记
    
    % 遍历每个簇，绘制簇内点
    for k = 1:4
        cluster_points = feature(c_indx == k, :); % 提取属于第 k 簇的点
        scatter(cluster_points(:, 1), cluster_points(:, 2), 50, colors(k), markers(k), 'filled');
    end
    
    % 绘制簇中心
    scatter(c_point(:, 1), c_point(:, 2), 100, 'k', 'x', 'LineWidth', 2, 'DisplayName', 'Cluster Center');
    
    % 图形修饰
    title('Clustering Results with k = 4');
    xlabel('Profit');
    ylabel('Weight');
    legend({'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster Centers'});
    grid on;
    hold off;
end