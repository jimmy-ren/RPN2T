function [labels ] = gen_norm_gen_labels(net, img, boxes, opts)

    n = size(boxes,1);
    ims = gen_norm_extract_regions(img, boxes, opts);
    nBatches = ceil(n/ opts.batch_size);

    for i=1:nBatches
    %     fprintf('extract batch %d/%d...\n',i,nBatches);

        batch = ims(:,:,:, opts.batch_size*(i-1)+1:min(end, opts.batch_size*i));
        % permute data into caffe c++ memory, thus [num, channels, height, width]
        batch = batch(:, :, [3, 2, 1], :); % from rgb to brg
        batch = permute(batch, [2, 1, 3, 4]);
        batch = single(batch);

        net_inputs = {batch};

        % Reshape net's input blobs
        net.reshape_as_input(net_inputs);
        res = net.forward(net_inputs);

        f = res{1};
        if ~exist('labels','var')
            labels = zeros(size(f,1),size(f,2),size(f,3),n,'single');
        end
            labels(:,:,:,opts.batch_size*(i-1)+1:min(end,opts.batch_size*i)) = f;
    end

end