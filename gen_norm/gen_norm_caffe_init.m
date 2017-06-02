function [labels_net] = gen_norm_caffe_init(opts)

    caffe.reset_all();
    caffe.set_mode_gpu();
    caffe.set_device(0);    
    
    labels_net = caffe.Net( opts.caffe_labels_gen_net, 'test');
    labels_net.copy_from(opts.caffe_initial_weights);
      
end