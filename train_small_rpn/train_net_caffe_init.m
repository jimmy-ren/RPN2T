function [labels_net, learn_net_solver] = train_net_caffe_init(opts)

    caffe.reset_all();
    caffe.set_mode_gpu();
    caffe.set_device(0);    
    
    labels_net = caffe.Net( opts.caffe_labels_gen_net, 'test');
    labels_net.copy_from(opts.caffe_initial_weights);
    
    learn_net_solver = caffe.Solver(opts.caffe_learn_net_solver);
    
    if ~exist(opts.save_path,'file')
       learn_net_solver.net.copy_from(opts.caffe_initial_weights)
    else
       learn_net_solver.net.copy_from(opts.save_path)
    end
   
end