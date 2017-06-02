function train_net_train_small_net(solver,im, boxes, labels, opts)
global loss;
num_boxes = size(boxes, 1);
crop_mode = opts.crop_mode;
crop_size = opts.learn_input_size;
crop_padding = opts.crop_padding;

ims = zeros(crop_size, crop_size, 3, num_boxes, 'single');
% mean_rgb = mean(mean(single(im)));

parfor i = 1:num_boxes
    bbox = boxes(i,:);
    crop = im_crop(im, bbox, crop_mode, crop_size, crop_padding);
    ims(:,:,:,i) = crop;
end



n = size(boxes,1);
nBatches = ceil(n/ opts.batch_size);

loss_=0;

for i=1:nBatches
        batch = ims(:,:,:, opts.batch_size*(i-1)+1:min(end, opts.batch_size*i));
        sublabels = labels(:,:,:, opts.batch_size*(i-1)+1:min(end, opts.batch_size*i));
        % permute data into caffe c++ memory, thus [num, channels, height, width]
        batch = batch(:, :, [3, 2, 1], :); % from rgb to brg
        batch = permute(batch, [2, 1, 3, 4]);
        batch = single(batch);
        labels_weight = ones(size(sublabels),'single');
        net_inputs = {batch,sublabels,labels_weight, opts.mean, opts.stds};

        % Reshape net's input blobs
        solver.net.reshape_as_input(net_inputs);
        solver.net.set_input_data(net_inputs);
        solver.step(1);
        loss_ =loss_+ solver.net.blobs('loss').get_data();
end
        loss=loss+loss_/nBatches;
end 
