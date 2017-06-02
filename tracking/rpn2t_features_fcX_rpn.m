function [ feat ] = rpn2t_features_fcX_rpn(solver, ims, opts)

n = size(ims,4);
nBatches = ceil(n/(opts.batchSize*2));

tic

for i=1:nBatches
    
    batch = ims(:,:,:,opts.batchSize*2*(i-1)+1:min(end,opts.batchSize*2*i));
    batch = single(batch);

    label_tmp = rand(size(batch,1), size(batch,2), 1, size(batch,4));
    weight_tmp = rand(size(label_tmp));
    %net_inputs = {batch, label_tmp, weight_tmp,label_tmp,weight_tmp};
    net_inputs = {batch, label_tmp, label_tmp};

    % Reshape net's input blobs
    %solver.net.reshape_as_input(net_inputs);
    solver.net.blobs('data').reshape(size(batch))
    solver.net.blobs('labels1').reshape(size(label_tmp))
    solver.net.blobs('labels2').reshape(size(label_tmp))
    solver.net.reshape();
    solver.net.forward(net_inputs);
    res1 = solver.net.blobs('proposal_cls_prob1').get_data();
    res1 = res1(:,:,1,:);
    res2 =  solver.net.blobs('proposal_cls_prob2').get_data();
    res2 = res2(:,:,1,:);
        
    sub_weight1 = opts.weight1.*opts.weight_mask1;
    sub_weight2 = opts.weight2.*opts.weight_mask2;
        
    res1 = res1.*repmat(sub_weight1, [1 1 1 size(res1,4)]);
    res1 = squeeze(sum(sum(res1))) / sum(sum(sub_weight1));
      
    res2 = res2.*repmat(sub_weight2, [1 1 1 size(res2,4)]);
    res2 = squeeze(sum(sum(res2))) / sum(sum(sub_weight2));
    %res = (res1+res2)/2;
    res = 0.45*res1 + 0.55*res2;
    
    if ~exist('feat','var')
        feat = zeros(1,n,'single');
    end
    feat(opts.batchSize*2*(i-1)+1:min(end,opts.batchSize*2*i)) = res;
    
end

spf = toc;
fprintf('time of score = %f\n',spf);