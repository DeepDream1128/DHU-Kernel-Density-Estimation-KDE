% 读取数据
filename = 'gaussian_3.618608.txt'; % 请将此处替换为您的数据文件名
data = importdata(filename);
w=2160;
h=3840;
% 检查数据尺寸是否正确
if size(data) ~= [w, h]
    disp('数据尺寸不符合要求。请检查数据。');
    return;
end

% 创建X、Y和Z网格
[x, y] = meshgrid(1:h, 1:w);

% 绘制三维热力图
% 归一化数据
%data=abs(data);


% 绘制三维热力图
figure;

surf(x, y, data)
view(3);
shading flat;
colormap(jet); 
colorbar;
xlabel('X轴');
ylabel('Y轴');
zlabel('Density');
title('三维热力图');
z_min = min(min(data));
z_max = max(max(data));
zlim([z_min,5*z_max]);

