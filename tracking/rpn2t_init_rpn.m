function [feat_extract_net, track_net_solver] = rpn2t_init_rpn(opts)

    caffe.reset_all();
    caffe.set_mode_gpu();
    caffe.set_device(0);    
    
    feat_extract_net = caffe.Net(opts.caffe_feat_extract_net, 'test');
    feat_extract_net.copy_from(opts.caffe_feat_extract_weights);
    
    track_net_solver = caffe.Solver(opts.caffe_track_net_solver);
    track_net_solver.net.copy_from(opts.caffe_tracker_init_weights);
end