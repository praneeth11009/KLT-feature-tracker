clear;
close all;
clc;

%% Your Code Here
noOfFeatures = 5; % Set this parameter to choose how many feature points are to be tracked
patchSize = 40; % Set this to change patch size
Thresold = 0.01;
maxiter = 30;
im_size = [480,640];

imagefiles = dir('../input/*.jpg');
N = length(imagefiles);

Frames = zeros(N,im_size(1),im_size(2));
trackedPoints = [];
for i = 1:N
    if i >= 60
        filename = strcat('../input/',num2str(i+1),'.jpg');
        Frames(i,:,:) = im2double(imread(filename));
    else 
        filename = strcat('../input/',num2str(i),'.jpg');
        Frames(i,:,:) = im2double(imread(filename));
    end
    
end
first_frame = reshape(Frames(1,:,:),im_size);
corners = detectHarrisFeatures(first_frame,'ROI',[115,180,125,130]);
features = corners.selectStrongest(noOfFeatures);
features = (features.Location)' ;

h = figure;
imshow(first_frame); hold on;
allfeatures = (corners.Location)';
plot(allfeatures(1,:),allfeatures(2,:),'*');
title('Major features in first frame (only a few of them are selected for tracking)');
hold off;
waitfor(h);

trackedPoints(1,:,:) = features;
prev_frame = first_frame;
for i = 2:N
    disp(['FRAME : ',num2str(i)]);
    next_frame = reshape(Frames(i,:,:),im_size);
    for f = 1:noOfFeatures 
        posx = trackedPoints(i-1,2,f);
        posy = trackedPoints(i-1,1,f);
        x1 = int32(posx-patchSize/2+1);%x1 = max(1,posx-patchSize/2+1);
        x2 = int32(posx+patchSize/2);%x2 = min(im_size(1),posx+patchSize/2);
        y1 = int32(posy-patchSize/2+1);%y1 = max(1,posy-patchSize/2+1);
        y2 = int32(posy+patchSize/2);%y2 = min(im_size(2),posy+patchSize/2);
        T = prev_frame(x1:x2,y1:y2);
        p0 = [1 0 posx 0 1 posy];
        RMSE = 100;
        for k = 1:maxiter
            if RMSE < Thresold
                break
            end
            W = [p0(1),p0(2),p0(3); p0(4),p0(5),p0(6)];
            I = zeros(size(T));
            for x_itr = 1:size(T,1) 
                for y_itr = 1:size(T,2)
                    orig = [x_itr-patchSize/2;y_itr-patchSize/2;1];
                    trans = W*orig;
                    x_trans = trans(1);% x_trans = max(1,x_trans); x_trans = min(x_trans,im_size(1));
                    y_trans = trans(2);% y_trans = max(1,y_trans); y_trans = min(y_trans,im_size(2));
                    I(x_itr,y_itr) = next_frame(int32(x_trans),int32(y_trans));
                end
            end
            [~,~,Gy,Gx] = edge(imgaussfilt(I,0.1),'sobel');
            %[~,~,Gy,Gx] = edge(I,'sobel');
            %Gx = imgaussfilt(Gx,1);
            %Gy = imgaussfilt(Gy,1);
            
            H = zeros(6,6);
            J = zeros(1,6);
            for x_itr = 1:size(I,1)
                for y_itr = 1:size(I,2)
                    gx_ij = Gx(x_itr,y_itr);
                    gy_ij = Gy(x_itr,y_itr);
                    x = x_itr - (patchSize/2);
                    y = y_itr - (patchSize/2);
                    temp = [x*gx_ij, y*gx_ij, gx_ij, x*gy_ij, y*gy_ij, gy_ij];
                    H = H + temp'*temp;
                    J = J + (T(x_itr,y_itr)-I(x_itr,y_itr))*temp;
                end
            end
            dp = J/H;
            p0 = p0 - dp;
            RMSE = sqrt(mean(mean((T - I).^2)));
        end
        disp(['RMSE for patch of feature ',num2str(f),' ',num2str(RMSE)]);
        new_x = p0(3);
        new_y = p0(6);
        trackedPoints(i,2,f) = new_x;
        trackedPoints(i,1,f) = new_y;
    end
   
    %if mod(i,30) == 0
    %    h = figure;
    %    imshow(next_frame); hold on;
    %    plot(reshape(trackedPoints(i,1,:),[1,noOfFeatures]),reshape(trackedPoints(i,2,:),[1,noOfFeatures]),'*');
    %    waitfor(h);
    %end
    prev_frame = next_frame;
end
%% Save all the trajectories frame by frame
% variable trackedPoints assumes that you have an array of size 
% No of frames * 2(x, y) * No Of Features
% noOfFeatures is the number of features you are tracking
% Frames is array of all the frames(assumes grayscale)

figure;
noOfPoints = 1;
for i=1:N
    NextFrame = reshape(Frames(i,:,:),im_size);
    imshow(NextFrame); hold on;
    for nF = 1:noOfFeatures
        plot(trackedPoints(1:noOfPoints, 1, nF), trackedPoints(1:noOfPoints, 2, nF),'*')
    end
    %hold off;
    saveas(gcf,strcat('../output/',num2str(i),'.jpg'));
    %close all;
    noOfPoints = noOfPoints + 1;
end 
