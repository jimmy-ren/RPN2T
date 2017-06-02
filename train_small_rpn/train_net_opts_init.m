function [opts] = train_net_opts_init


    opts.caffe_labels_gen_net = 'models/gen_norm_models/proposal_ZF_label.prototxt';
    
    %opts.caffe_initial_weights = 'models/rpn/proposal_final_ZF';
    opts.caffe_initial_weights = 'models/rpn/ZF_iter_160000';
    
    opts.caffe_learn_net_solver = 'models/gen_norm_models/solver.prototxt';
    
    opts.save_path = 'models/rpn/proposal_learn_160k_voc';
  
    opts.batch_size = 32; % <- reduce it in case of out of gpu memory

 
    opts.input_size = 203;
    opts.learn_input_size = 107;
    
    opts.crop_mode = 'wrap';
    opts.crop_largegt = false;
    opts.crop_padding = 16;
    
    opts.nsamples_per_frame = 256;
    opts.scale_factor = 1.05;
    
    opts.max_itr = 100000;
    
    
    load('Ex_160k_voc.mat');
    load('stds_160k_voc.mat');
    opts.mean = -repmat(Ex,[1, 1, 1, opts.batch_size]);
    opts.stds = bsxfun(@rdivide,1,repmat(stds,[1, 1, 1, opts.batch_size]));

end
