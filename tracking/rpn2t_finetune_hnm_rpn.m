function [poss, hardnegs] = rpn2t_finetune_hnm_rpn(solver,pos_data,neg_data,opts,maxiter)

 opts.useGpu = true;
 opts.conserveMemory = true ;
 opts.sync = true ;
% 
 opts.maxiter = maxiter;
 opts.learningRate = 0.001;
 opts.weightDecay = 0.0005 ;
 opts.momentum = 0.9 ;
% 
opts.batchSize_hnm = 256;
opts.batchAcc_hnm = 4;
% 
% % opts.batchSize = 128;
% % opts.batch_pos = 32;
% % opts.batch_neg = 96;
% 
% opts = vl_argparse(opts, varargin) ;



res = [] ;

n_pos = size(pos_data,4);
n_neg = size(neg_data,4);
train_pos_cnt = 0;
train_neg_cnt = 0;

% extract positive batches
train_pos = [];
remain = opts.batch_pos*opts.maxiter;
while(remain>0)
    if(train_pos_cnt==0)
        train_pos_list = randperm(n_pos)';
    end
    train_pos = cat(1,train_pos,...
        train_pos_list(train_pos_cnt+1:min(end,train_pos_cnt+remain)));
    train_pos_cnt = min(length(train_pos_list),train_pos_cnt+remain);
    train_pos_cnt = mod(train_pos_cnt,length(train_pos_list));
    remain = opts.batch_pos*opts.maxiter-length(train_pos);
end

% extract negative batches
train_neg = [];
remain = opts.batchSize_hnm*opts.batchAcc_hnm*opts.maxiter;
while(remain>0)
    if(train_neg_cnt==0)
        train_neg_list = randperm(n_neg)';
    end
    train_neg = cat(1,train_neg,...
        train_neg_list(train_neg_cnt+1:min(end,train_neg_cnt+remain)));
    train_neg_cnt = min(length(train_neg_list),train_neg_cnt+remain);
    train_neg_cnt = mod(train_neg_cnt,length(train_neg_list));
    remain = opts.batchSize_hnm*opts.batchAcc_hnm*opts.maxiter-length(train_neg);
end

% learning rate
lr = opts.learningRate ;

% for saving positives
poss = [];

% for saving hard negatives
hardnegs = [];

% objective fuction
objective = zeros(1,opts.maxiter);

%% training on training set
% fprintf('\n');
for t=1:opts.maxiter
%     fprintf('\ttraining batch %3d of %3d ... ', t, opts.maxiter) ;
    
    % ----------------------------------------------------------------------
    % hard negative mining
    % ----------------------------------------------------------------------
    score_hneg = zeros(opts.batchSize_hnm*opts.batchAcc_hnm,1);
    hneg_start = opts.batchSize_hnm*opts.batchAcc_hnm*(t-1);
    for h=1:opts.batchAcc_hnm
        batch = neg_data(:,:,:,...
            train_neg(hneg_start+(h-1)*opts.batchSize_hnm+1:hneg_start+h*opts.batchSize_hnm));        
        
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
        score_hneg((h-1)*opts.batchSize_hnm+1:h*opts.batchSize_hnm) = res;
    end
    [~,ord] = sort(score_hneg,'descend');
    hnegs = train_neg(hneg_start+ord(1:opts.batch_neg));
    im_hneg = neg_data(:,:,:,hnegs);
%     fprintf('hnm: %d/%d, ', opts.batch_neg, opts.batchSize_hnm*opts.batchAcc_hnm) ;
    hardnegs = [hardnegs; hnegs];
    
    % ----------------------------------------------------------------------
    % get next image batch and labels
    % ----------------------------------------------------------------------
    poss = [poss; train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos)];
    
    batch = cat(4,pos_data(:,:,:,train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos)),...
        im_hneg);
    
    labels1 = ones(size(batch,1), size(batch,2), 1, size(batch,4));
    labels1(:,:,:,1:opts.batch_pos) = repmat(not(opts.weight_mask1),[1 1 1 opts.batch_pos]);
    
    label_weights1 = ones(size(labels1));
    label_weights1(:,:,:,1:opts.batch_pos) = repmat(sub_weight1,[1 1 1 opts.batch_pos]);
    label_weights1(:,:,:,opts.batch_pos+1:end) = repmat(opts.weight_mask1,[1 1 1 (size(labels1,4)-opts.batch_pos)]);
   
    labels2 = ones(size(labels1));
    labels2(:,:,:,1:opts.batch_pos)  = repmat(not(opts.weight_mask2),[1 1 1 opts.batch_pos]);
    label_weights2 = repmat( opts.weight_mask2 ,[1 1 1 (size(labels2,4))]);
   
    
    %net_inputs = {batch, labels1 , label_weights1 , labels2 , label_weights2};    
    %net_inputs = {batch, labels1 , label_weights1 , labels1 , label_weights1};
    %solver.net.reshape_as_input(net_inputs);
    labels1(label_weights1 == 0) = -1;
    labels2(label_weights2 == 0) = -1;
    solver.net.blobs('data').reshape(size(batch));
    solver.net.blobs('labels1').reshape(size(labels1))
    solver.net.blobs('labels2').reshape(size(labels2))
    solver.net.reshape();

    % one iter SGD update
    %solver.net.set_input_data(net_inputs);
    solver.net.blobs('data').set_data(batch);
    solver.net.blobs('labels1').set_data(labels1);
    solver.net.blobs('labels2').set_data(labels2);
    solver.step(1);
end % next batch


end