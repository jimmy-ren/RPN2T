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

root_dir = [pwd, '/dataset/VOT/2016/'];
sub_dirs = dir(root_dir);
total_video = length(sub_dirs);
opts = gen_norm_opts_init;
[labels_net] = gen_norm_caffe_init(opts);


 counts = zeros(1,1)+eps;

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
    

   
   if ~mod(itr,100)
      fprintf('saving...');
      Ex = bsxfun(@rdivide, sumEx, counts);
      stds = (bsxfun(@minus, bsxfun(@rdivide, sumEx2, counts), Ex.^2)).^0.5;
      save('Ex','Ex');
      save('stds','stds');
   end
 
end
  
   
   
   
   
   
   
   
   
   
