function [opts] = rpn2t_init(image)

    %% set opts
    % use gpu
    %opts.useGpu = true;

    % model def
    %opts.net_file = net;
    global versions;
    global nets;
    
    if     versions==1
        if nets==1
            opts.caffe_feat_extract_net = 'models/rpn/proposal_test_init_ZF.prototxt';
            %opts.caffe_feat_extract_weights = 'models/rpn/proposal_final_ZF';
            opts.caffe_feat_extract_weights = 'models/rpn/ZF_iter_160000';
        elseif nets==2
            opts.caffe_feat_extract_net = 'models/rpn/proposal_test_init_VGG.prototxt';
            opts.caffe_feat_extract_weights = 'models/rpn/proposal_final_VGG';
        end
       
    elseif versions==2
      opts.caffe_feat_extract_net = 'models/rpn/proposal_test_learn_1_pooling.prototxt';
      opts.caffe_feat_extract_weights = 'models/rpn/proposal_learn_1_pooling';
      
    elseif versions==3
      opts.caffe_feat_extract_net = 'models/rpn/proposal_test_learn_0_pooling.prototxt';
      %opts.caffe_feat_extract_weights = 'models/rpn/proposal_learn_0_pooling';
      opts.caffe_feat_extract_weights = 'models/rpn/proposal_learn_160k_voc_5_9000'; %proposal_learn_160k_voc_5_9000 %proposal_learn_160k_voc_3_9000
    elseif versions==4
      opts.caffe_feat_extract_net = 'models/rpn/proposal_test_learn_0_norm.prototxt';
      opts.caffe_feat_extract_weights = 'models/rpn/proposal_learn_0_norm';        
    end
    
    
    if nets==1
       %opts.caffe_tracker_init_weights = 'models/rpn/proposal_final_ZF';
       opts.caffe_tracker_init_weights = 'models/rpn/ZF_iter_160000';
       opts.caffe_track_net_solver = 'models/tracking/solver_ZF.prototxt';
    elseif nets==2
       opts.caffe_tracker_init_weights = 'models/rpn/proposal_final_VGG';
       opts.caffe_track_net_solver = 'models/tracking/solver_VGG.prototxt';
    end
       


    % test policy
    opts.batchSize_test = 256; % <- reduce it in case of out of gpu memory

    % bounding box regression
    opts.bbreg = true;
    opts.bbreg_nSamples = 1000;
    
  % cropping policy
    if     versions==1
       if nets==1
          opts.input_size = 203;
       elseif nets==2
          opts.input_size = 210;
       end
        

    elseif versions==2||versions==3||versions==4
       opts.input_size = 107;
    end
    
    opts.crop_mode = 'wrap';
    opts.crop_largegt = false;
    opts.crop_padding = 16;
    % learning policy
    %% 设计label_weight和mask
%    load('output_height_map.mat');
%    load('output_width_map.mat');
%    
   feature_map_H = 14;%output_height_map([opts.input_size]);
   feature_map_W = 14;%output_width_map([opts.input_size]);
   
   if(not(mod(feature_map_H,2)&&mod(feature_map_W,2)))
       subweight1 = ((171-16*(floor(feature_map_W/2)-1)):16:171)';     
   end
   
   subweight1=repmat(subweight1,[1,floor(feature_map_W/2)]);
   subweight1 = subweight1.*subweight1';
   weight1 = [subweight1,flip(subweight1,2);flip(subweight1,1),flip(flip(subweight1,1),2)];
   %weight_12_12 = weight_12_12./(max(max(weight_12_12)));
   weight1 = weight1./(max(max(weight1))*2-weight1);
   weight_mask1 = weight1>0.7;
    %%    
    opts.weight1 = weight1;
    opts.weight_mask1 = weight_mask1;
    opts.weight2 = zeros(feature_map_H,feature_map_W);
    opts.weight2(3:12,3:12) = 1;
    opts.weight_mask2 = opts.weight2;
    
    %%
    opts.batchSize = 128;
    opts.batch_pos = 32;
    opts.batch_neg = 96;
    
    
    

    % initial training policy
    opts.learningRate_init = 0.0001; % x10 for fc6
    opts.maxiter_init = 30;%30;

    opts.nPos_init = 500;%500;
    opts.nNeg_init = 5000;
    opts.posThr_init = 0.6;
    opts.negThr_init = 0.5;

    % update policy
    opts.learningRate_update = 0.0003; % x10 for fc6
    opts.maxiter_update = 10; %10;

    opts.nPos_update = 25;%100;
    opts.nNeg_update = 100;%300;
 
    opts.posThr_update = 0.7;
    opts.negThr_update = 0.3;

    opts.update_interval = 10; % interval for long-term update

    % data gathering policy
    opts.nFrames_long = 100; % long-term period
    opts.nFrames_short = 20; % short-term period

    % scaling policy
    opts.scale_factor = 1.05;

    % sampling policy
    opts.nSamples = 256;
    opts.trans_f = 0.6; % translation std: mean(width,height)*trans_f/2
    opts.scale_f = 1; % scaling std: scale_factor^(scale_f/2)

    % set image size
    opts.imgSize = size(image);
end
