main_folder = 'D:\siyi_vessle\ji_amir_resize\';
subjects = dir(main_folder);
for jj = 3:length(subjects)
    folder = [main_folder,subjects(jj).name,'\branch\'];
    disp(folder)
%
% subject = num2str(10);
% path = [folder,subject,'\'];
% a = [path,'*.png'];
Files=dir([folder,'*.tif']);
for k=1:length(Files)
    FileNames=[folder, Files(k).name];
    I = imread(FileNames);    
    if ndims(I) == 3
        I = rgb2gray(I);
    end
    if length(I(I<=0)) > length(I(I>0))
        skeleton = bwskel(logical(I));
    else
        skeleton = bwskel(~logical(I));
    end
%     figure(1)
%     imshow(skeleton)
%     kernel = [1,1,1;1,1,1;1,1,1];
%     map = skeleton .* imfilter(skeleton, kernel,'same');
%     [row_sp,col_sp] = find(map == 2);
    [row,col] = find(skeleton > 0);
    c = polyfit(row,col,1);
    direction_vector = [size(I,1), c(1) * size(I,1) + c(2)] - [0, c(1) * 0 + c(2)];
    distance = zeros(length(row),1);
    for i = 1:length(row)
        vector = [row(i),col(i)] - [0, c(1) * 0 + c(2)];
        distance(i) = dot(vector, direction_vector);
    end
    [sorted_distance, idx] = sort(distance);
    sorted_row = row(idx);
    sorted_col = col(idx);
%     Display = zeros(size(skeleton));
% 
% %     for i = 1:size(I,1)
% %         j = round(i * c(1) + c(2));
% %         if j <= size(I,2) & j >= 1
% %             Display(i,round(i * c(1) + c(2))) = 100;
% %     
% %         end
% %     end
% 
%     for i = 1:length(row)
% %         Display(sorted_row(i),sorted_col(i)) = i;
%         Display(sorted_row(i),sorted_col(i)) = 1;
%     end
%     figure(1)
% 
%     imshow(Display,[])
%     title(k)

% % 1
%     if ismember(k,[2 6 7 8 14 15 16 19 20 21 22 24 25 26])
% % 2
%     if ismember(k,[1 2 4 5 15 17 18 19 20 21 25])
% % 3
%     if ismember(k,[9 13 14 15 16 20 22])
% % 4
%     if ismember(k,[1 2 3 4 10 11 12 15 16 17 18 19 20])
% % 5
%     if ismember(k,[1 2 4 10 13 14 15 19 20 24 25 27 29 30 31])
% % 6
%     if ismember(k,[1 2 7 8 11 19 20 21 23 24 25])
% % 7
%     if ismember(k,[2 3 4 6 9 10 12 13 15 18 20  25 26 28 29])
% % 8
%     if ismember(k,[1 2 5 6 7 10 11 16 17 19 20 21 24 25 26])
% % 9
%     if ismember(k,[1 2 4 5 6 7 12 13 14 15 19 20 25])
% % 10
%     if ismember(k,[4 5 7 8 9 13 14 17 20 23 24 25 26 27])
% % 11
%     if ismember(k,[2 3 4 5 6 9 11 14 15 19 20 21 25 26 27 28])
% % 12
%     if ismember(k,[4 7 9 10 14 15 17 20 21 26 27 29 32 33 38])
% % 13
%     if ismember(k,[2 4 7 8 9 10 11 12 16 19 21 22 23 24 25 27 29 30 31 34 35])
% % 14 
%     if ismember(k,[1 2 3  9 10 11 13 14 18 19 21 26 27 28 29])
% % 15
%     if ismember(k,[2 3 5 6 7 8 11 18 19 20 21 24 25 26 27 28])
% % 16
%     if ismember(k,[1 2 3 4 10 14 15 16 17 26 27 28 29 30])
% % 17
%     if ismember(k,[2 3 6 7 10 11 13 17 18 23 24 25 28 29])
% % 18
%     if ismember(k,[4 5 7 8 9 16 17 18 19 20 21 22 29 30])
% % 19
%     if ismember(k,[3 4 6 10 11 12 18 19 20 22 24 26 29 31 32 33])
% % 20
%     if ismember(k,[3 7 8 12 14 19 21 22 25 29 30 33 34])



%   old ----------------
%     10
%     if ismember(k,[2,5,6,8,9,10,11,15,16,19,22,25,26,27,28])
%     11
%     if ismember(k,[2,3,4,5,6,7,12,13,14,15,16,18,22,27,34,35,42,44,47,48,49,50,51])    
%     12
%     if ismember(k,[4,5,6,9,10,12,13,14,15,16,17,18,22,24,25,26,27,29,33,34,42,43,45,46,51,52,54,55,56,57,59,64])    

%     21
%     if ismember(k,[2,3,6,7,10,14,15,16,17,18,19])
%     22
%     if ismember(k,[1,2,3,5,6,9,10,11,13,16,17,18,19])
%     23
%     if ismember(k,[2,3,4,5,6,7,8,11,13])
%     24
%     if ismember(k,[1,3,4,5,9,12,13,14,16,21,24,25,30,31,35,36,37,41])
%     25
%     if ismember(k,[1,2,4,10,13,16,17,18,21,22,23,24,25,26,31])
%     26
%     if ismember(k,[4,5,6,7,8,13,14,16,17,18,19,20,21,23,24,25,27])
%     27
%     if ismember(k,[2,3,4,12,13,14,15,25,26,27,28,29,31,32,36,37,38])
%     28
%     if ismember(k,[1,2,6,7,11,12,15,16,18,19,20,21,22,37,38])

%   new ----------------
%     29
%     if ismember(k,[1,11,12,15,16,18])
%     30
%     if ismember(k,[1,6,7,8,12,13,14,17,18])
%     31
%     if ismember(k,[1,2,3,4,10,15,16,17,18,19])
%     32
%     if ismember(k,[1,6,9,10,13,14,15,16,17,18])
%     33
%     if ismember(k,[])//

%   old ----------------
%     29
%     if ismember(k,[1,12,14,15,18,19,20,22])
%     30
%     if ismember(k,[1,3,7,8,9,14,16,17,20,22])
%     31
%     if ismember(k,[1,2,3,4,5,13,20,21,22,23,24,25])
%     32
%     if ismember(k,[1,2,4,10,14,15,16,20,21,22,23,24,25])
%     33
%     if ismember(k,[1,2,3,4,5,8,9,10,11,12,13,15,20,22,23,24])

%     35
%     if ismember(k,[1,2,3,4,7,8,10,11,14,17,18,22,30,37,38,39])
%     36
%     if ismember(k,[1,3,5,9,10,11,12,13,19,20,22,23,24,25,26,28,29,32,33])
%     37
%     if ismember(k,[3,4,5,6,9,11,17,18,19])
%     38
%     if ismember(k,[1,2,3,12,13,14,17,18,19,20,21,24,25,29])
%     39
%     if ismember(k,[4,6,7,8,9,14,15,16,17,19,20])
%     40
%     if ismember(k,[1,3,5,6,7,9,10,11,12,13,14,15,16,20,22,24])

%         sorted_row = flip(sorted_row);
%         sorted_col = flip(sorted_col);
%     end
    writematrix([sorted_row';sorted_col'],[folder,'trace.txt'],'WriteMode','append','Delimiter','space')

 end

%print('down');
end