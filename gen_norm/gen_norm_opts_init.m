function [opts] = gen_norm_opts_init


    opts.caffe_labels_gen_net = 'models/gen_norm_models/proposal_ZF_init.prototxt';
    
    %opts.caffe_initial_weights = 'models/rpn/proposal_final_ZF';
    opts.caffe_initial_weights = 'models/rpn/ZF_iter_160000';
   
    opts.batch_size = 256; % <- reduce it in case of out of gpu memory

 
    opts.input_size = 203;
    opts.learn_input_size = 107;
    
    opts.crop_mode = 'wrap';
    opts.crop_largegt = false;
    opts.crop_padding = 16;
    
    opts.nsamples_per_frame = 256;
    opts.scale_factor = 1.05;
    
    opts.max_itr = 1000;

end
