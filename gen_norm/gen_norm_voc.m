clear;

addpath utils/
addpath utils/rpn/
addpath external/caffe/matlab/
addpath tracking/
addpath gen_norm/

if exist('Ex.mat','file')&& exist('stds.mat','file')
   warning('Ex and stds have already existed') 
end

if(isempty(gcp('nocreate')))
    parpool;
end

xml_root_dir = '/media/jimmyren/Work3/data/VOC/VOCdevkit/VOC2012/Annotations/';
img_root_dir = '/media/jimmyren/Work3/data/VOC/VOCdevkit/VOC2012/JPEGImages/';
opts = gen_norm_opts_init;
[labels_net] = gen_norm_caffe_init(opts);
counts = zeros(1,1)+eps;

xml_list = dir(strcat(xml_root_dir, '*.xml'));
idx = randperm(length(xml_list));
xml_list = xml_list(idx);

for file = 1:length(xml_list)

    fprintf('processing file %d/%d\n', file, length(xml_list));    
    xml_struct = xml2struct(strcat(xml_root_dir, xml_list(file).name));    
    img = imread(strrep(strcat(img_root_dir, xml_list(file).name), '.xml', '.jpg'));
    if(size(img,3)==1)
        img = cat(3,img,img,img);
    end
    opts.imgSize = size(img);
    
    obj_num = length(xml_struct.annotation.object);
    obj_idx = randi(obj_num);
    
    if(obj_num > 1)
        targetLoc = [str2double(xml_struct.annotation.object{obj_idx}.bndbox.xmin.Text) ...
                     str2double(xml_struct.annotation.object{obj_idx}.bndbox.ymin.Text) ...
                     str2double(xml_struct.annotation.object{obj_idx}.bndbox.xmax.Text) - str2double(xml_struct.annotation.object{obj_idx}.bndbox.xmin.Text) ...
                     str2double(xml_struct.annotation.object{obj_idx}.bndbox.ymax.Text) - str2double(xml_struct.annotation.object{obj_idx}.bndbox.ymin.Text)];
    else
        targetLoc = [str2double(xml_struct.annotation.object.bndbox.xmin.Text) ...
                     str2double(xml_struct.annotation.object.bndbox.ymin.Text) ...
                     str2double(xml_struct.annotation.object.bndbox.xmax.Text) - str2double(xml_struct.annotation.object.bndbox.xmin.Text) ...
                     str2double(xml_struct.annotation.object.bndbox.ymax.Text) - str2double(xml_struct.annotation.object.bndbox.ymin.Text)];
    end


    

    %% Extract training examples   
    Samples1 = gen_samples('gaussian', targetLoc, opts.nsamples_per_frame/2, opts, 0.1, 5);
    Samples2 = gen_samples('gaussian', targetLoc, opts.nsamples_per_frame/2, opts, 1, 5);
    Samples = cat(1, Samples1, Samples2);
    
    Samples = Samples(randsample(end,opts.nsamples_per_frame),:);
    
%     imshow(img)
%     for a = 1:opts.nsamples_per_frame
%         rectangle('Position', Samples(a,:), 'EdgeColor', [1 0 0], 'Linewidth', 1);
%     end
%     pause;
    
    
    [labels] = gen_norm_gen_labels(labels_net, img, Samples, opts);
   
    if ~exist('sumEx','var')
        sumEx = zeros(size(labels,1),size(labels,2),size(labels,3),'single');
    end
    
    if ~exist('sumEx2','var')
        sumEx2 = zeros(size(sumEx),'single');
    end
    
    
    sumEx = bsxfun(@plus, sumEx,bsxfun(@rdivide,sum(labels,4),size(labels,4)));
    sumEx2 = bsxfun(@plus, sumEx2,bsxfun(@rdivide,sum(labels.^2,4),size(labels,4))) ;
    counts = counts+1;
    

   
   if ~mod(file,100)
      fprintf('saving...');
      Ex = bsxfun(@rdivide, sumEx, counts);
      stds = (bsxfun(@minus, bsxfun(@rdivide, sumEx2, counts), Ex.^2)).^0.5;
      save('Ex','Ex');
      save('stds','stds');
   end
 
end
  
   
   
   
   
   
   
   
   
   
