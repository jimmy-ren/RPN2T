function [ feat ] = rpn2t_features_convX_rpn(net, img, boxes, opts)

tic
    n = size(boxes,1);
    ims = rpn2t_extract_regions(img, boxes, opts);
    nBatches = ceil(n/opts.batchSize_test);
    
spf = toc;
fprintf('time of extract region = %f  ',spf);

tic
    for i=1:nBatches
    %     fprintf('extract batch %d/%d...\n',i,nBatches);

        batch = ims(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i));
        % permute data into caffe c++ memory, thus [num, channels, height, width]
        batch = batch(:, :, [3, 2, 1], :); % from rgb to brg
        batch = permute(batch, [2, 1, 3, 4]);
        batch = single(batch);

        net_inputs = {batch};

        % Reshape net's input blobs
        %net.reshape_as_input(net_inputs);
        net.blobs('data').reshape(size(batch));
        net.reshape();
        res = net.forward(net_inputs);

        f = res{1};
        if ~exist('feat','var')
            feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
        end
        feat(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i)) = f;
    end
    
spf = toc;
fprintf('time of forward = %f\n',spf);

end