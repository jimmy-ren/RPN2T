clear;

addpath utils/
addpath utils/rpn/
addpath external/caffe/matlab/
addpath tracking/
addpath train_small_rpn/

global loss;
loss = zeros(1,1)+eps; 
try 
   load('saveloss.mat'); 
catch 
   saveloss = []; 
end

if(isempty(gcp('nocreate')))
    parpool;
end

xml_root_dir = '/media/jimmyren/Work3/data/VOC/VOCdevkit/VOC2012/Annotations/';
img_root_dir = '/media/jimmyren/Work3/data/VOC/VOCdevkit/VOC2012/JPEGImages/';
opts = train_net_opts_init;
[labels_net, learn_net_solver] = train_net_caffe_init(opts);
counts = zeros(1,1)+eps;

xml_list = dir(strcat(xml_root_dir, '*.xml'));
idx = randperm(length(xml_list));
xml_list = xml_list(idx);

for pass = 1:5
    idx = randperm(length(xml_list));
    xml_list = xml_list(idx);
    
    for file = 1:length(xml_list)

        fprintf('pass %d, processing file %d/%d\n', pass, file, length(xml_list));    
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

    %     imshow(img)
    %     for a = 1:opts.nsamples_per_frame
    %         rectangle('Position', Samples(a,:), 'EdgeColor', [1 0 0], 'Linewidth', 1);
    %     end
    %     pause;

        [labels] = train_net_gen_labels(labels_net, img, Samples, opts);
       %% Train small ZF network
       train_net_train_small_net(learn_net_solver,img, Samples, labels, opts);

       if ~mod(file,100)
           saveloss = [saveloss,loss/100];
           save('saveloss','saveloss');
           loss = zeros(1,1)+eps;
       end

       if ~mod(file,1000)
           learn_net_solver.net.save(strcat(opts.save_path, '_', num2str(pass), '_', num2str(file)));
       end    
    end
end
