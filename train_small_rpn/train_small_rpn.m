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

root_dir = [pwd, '/dataset/VOT/2016/'];
sub_dirs = dir(root_dir);
total_video = length(sub_dirs);
opts = train_net_opts_init;
[labels_net, learn_net_solver] = train_net_caffe_init(opts);

for itr = 1:opts.max_itr 
    
    i = randperm(total_video-2,1)+2;
    
    if not(sub_dirs(i).isdir)
        i = i+1;
    end

 videoname = sub_dirs(i).name;
 fprintf(['itr = ',num2str(itr),' ','Preparing samples and labels of No.',num2str(i),' ',videoname,'\n']);
 conf = genConfig('vot2016',videoname);
 
 images = conf.imgList;
 region = conf.gt(1,:);


    %% Initialization

    nFrames = length(images);
    
    frame = randperm(nFrames,1);
    
    img = imread(images{frame});
    if(size(img,3)==1), img = cat(3,img,img,img); end
    opts.imgSize = size(img);
    targetLoc = region;

    

    

    %% Extract training examples
   
    Samples = gen_samples('uniform', targetLoc, opts.nsamples_per_frame, opts, 2, 5);
    Samples = Samples(randsample(end,opts.nsamples_per_frame),:);
    [labels] = train_net_gen_labels(labels_net, img, Samples, opts);
   %% Train small ZF network
   fprintf('Training\n');
   train_net_train_small_net(learn_net_solver,img, Samples, labels, opts);
   
   
   if ~mod(itr,100)
   learn_net_solver.net.save(opts.save_path)
   saveloss = [saveloss,loss/100];
   save('saveloss','saveloss');
   loss = zeros(1,1)+eps;
   end
    
end
