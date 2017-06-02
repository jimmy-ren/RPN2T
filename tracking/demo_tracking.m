
clear;

addpath utils/
addpath utils/rpn/
addpath tracking/
addpath external/_caffe/matlab/

if(isempty(gcp('nocreate')))
    parpool;
end

%% 
%1 original rpn
%2 simplified 1 pooling 1 norm
%3 simplified 0 pooling 1 norm
%4 0 pooling 1 norm 
global versions;

versions = 3;

%net 1 ZF
%net 2 VGG
global nets;
nets = 1;

%%
root_dir = [pwd, '/dataset/VOT/2016/'];
sub_dirs = dir(root_dir);
total = length(sub_dirs);
for i = 3:total 
    if not(sub_dirs(i).isdir)
        continue;
    end
  video = sub_dirs(i).name;
clc
%%
conf = genConfig('vot2016',video);

result = rpn2t_run_rpn(video,conf.imgList, conf.gt(1,:),true);

end
