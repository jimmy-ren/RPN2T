function [ result ] = rpn2t_run_rpn(videoname,images, region, display)

    if(nargin<3), display = true; end

    %% Initialization
    fprintf('Initialization...\n');

    nFrames = length(images);

    img = imread(images{1});
    if(size(img,3)==1), img = cat(3,img,img,img); end
    targetLoc = region;
    result = zeros(nFrames, 4); result(1,:) = targetLoc;

    opts = rpn2t_init(img);
    [feat_extract_net, track_net_solver] = rpn2t_init_rpn(opts);

    %% Train a bbox regressor
    if(opts.bbreg)
        pos_examples = gen_samples('uniform_aspect', targetLoc, opts.bbreg_nSamples*10, opts, 0.3, 10);
        r = overlap_ratio(pos_examples,targetLoc);
        pos_examples = pos_examples(r>0.6,:);
        pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);
        feat_conv = rpn2t_features_convX_rpn(feat_extract_net, img, pos_examples, opts);

        X = permute(feat_conv,[4,3,1,2]);
        X1 = X(:,:,7:8,6:9);
        X2 = X(:,:,6:6,7:8);
        X3 = X(:,:,9:9,7:8);
        X1 = X1(:,:);
        X2 = X2(:,:);
        X3 = X3(:,:);
        X = cat(2, X1, X2, X3);

%         X1 = X(:,:,3,3); X2 = X(:,:,3,6); X3 = X(:,:,3,9); X4 = X(:,:,3,12);
%         X5 = X(:,:,6,3); X6 = X(:,:,6,6); X7 = X(:,:,6,9); X8 = X(:,:,6,12);        
%         X9 = X(:,:,9,3); X10 = X(:,:,9,6); X11 = X(:,:,9,9); X12 = X(:,:,9,12);
%         X13 = X(:,:,12,3); X14 = X(:,:,12,6); X15 = X(:,:,12,9); X16 = X(:,:,12,12);        
%         
%         X1 = X1(:,:); X2 = X2(:,:); X3 = X3(:,:); X4 = X4(:,:);
%         X5 = X5(:,:); X6 = X6(:,:); X7 = X7(:,:); X8 = X8(:,:);        
%         X9 = X9(:,:); X10 = X10(:,:); X11 = X11(:,:); X12 = X12(:,:);
%         X13 = X13(:,:); X14 = X14(:,:); X15 = X15(:,:); X16 = X16(:,:);
%         X = cat(2, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16);
        
        bbox = pos_examples;
        bbox_gt = repmat(targetLoc,size(pos_examples,1),1);
        bbox_reg = train_bbox_regressor(X, bbox, bbox_gt);
    end

    %% Extract training examples
    fprintf('  extract features...\n');
spf1 = tic;

    % draw positive/negative samples
    pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_init*2, opts, 0.1, 5);
    r = overlap_ratio(pos_examples,targetLoc);
    pos_examples = pos_examples(r>opts.posThr_init,:);
    pos_examples = pos_examples(randsample(end,min(opts.nPos_init,end)),:);

    neg_examples = [gen_samples('uniform', targetLoc, opts.nNeg_init, opts, 1, 10);...
        gen_samples('whole', targetLoc, opts.nNeg_init, opts)];
    r = overlap_ratio(neg_examples,targetLoc);
    neg_examples = neg_examples(r<opts.negThr_init,:);
    neg_examples = neg_examples(randsample(end,min(opts.nNeg_init,end)),:);

    examples = [pos_examples; neg_examples];
    pos_idx = 1:size(pos_examples,1);
    neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

    % to bigger crops
    if(opts.crop_largegt)
        examples = loc2bigloc(examples);
    end

    % extract features    
    feat_conv = rpn2t_features_convX_rpn(feat_extract_net, img, examples, opts);
    pos_data = feat_conv(:,:,:,pos_idx);
    neg_data = feat_conv(:,:,:,neg_idx);


    %% Learning CNN
    fprintf('  training cnn...\n');
    rpn2t_finetune_hnm_rpn(track_net_solver, pos_data, neg_data, opts,opts.maxiter_init);

spf1 = toc(spf1);
    %% Initialize displayots
    if display
        figure(2);
        set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');

        hd = imshow(img,'initialmagnification','fit'); hold on;
        rectangle('Position', targetLoc, 'EdgeColor', [1 0 0], 'Linewidth', 1);
        set(gca,'position',[0 0 1 1]);

        text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
        
        %% save
        img_save = getframe(gca);
        mkdir(['output/','save_',videoname]);
        imwrite(img_save.cdata, strcat(['output/','save_',videoname,'/'], num2str(1), '.png'));
    
        %%
        hold off;
        drawnow;
    end

    %% Prepare training data for online update
    total_pos_data = cell(1,1,1,nFrames);
    total_neg_data = cell(1,1,1,nFrames);

    neg_examples = gen_samples('uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5);
    r = overlap_ratio(neg_examples,targetLoc);
    neg_examples = neg_examples(r<opts.negThr_init,:);
    neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);

    examples = [pos_examples; neg_examples];
    % to bigger crops
    if(opts.crop_largegt)
        examples = loc2bigloc(examples);
    end
    
    pos_idx = 1:size(pos_examples,1);
    neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);
    
    feat_conv = rpn2t_features_convX_rpn(feat_extract_net, img, examples, opts);
    total_pos_data{1} = feat_conv(:,:,:,pos_idx);
    total_neg_data{1} = feat_conv(:,:,:,neg_idx);
    total_pos_data{1} = permute(total_pos_data{1}, [1 4 3 2]);
    total_neg_data{1} = permute(total_neg_data{1}, [1 4 3 2]);

    success_frames = 1;
    trans_f = opts.trans_f;
    scale_f = opts.scale_f;

    %% Main loop
    for To = 2:nFrames;
        fprintf('\nProcessing frame %d/%d... \n', To, nFrames);

        img = imread(images{To});
        if(size(img,3)==1), img = cat(3,img,img,img); end

        spf = tic;
        %% Estimation
        % draw target candidates
        
        samples = gen_samples('gaussian', targetLoc, opts.nSamples, opts, trans_f, scale_f);

        % to bigger crops
        if(opts.crop_largegt)
            examples_big = loc2bigloc(samples);        
            feat_conv = rpn2t_features_convX_rpn(feat_extract_net, img, examples_big, opts);
        else
            feat_conv = rpn2t_features_convX_rpn(feat_extract_net, img, samples, opts);
        end

        % evaluate the candidates        
        feat_fc = rpn2t_features_fcX_rpn(track_net_solver, feat_conv, opts);

        feat_fc = feat_fc';
        [scores,idx] = sort(feat_fc,'descend');

        target_score = mean(scores(1:5));
        targetLoc = round(mean(samples(idx(1:5),:)));
        if(opts.crop_largegt)
            targetLoc_big = round(mean(examples_big(idx(1:5),:)));
        end
        
        if(To <= 50)
            score_thres = 0.2;
        else
            score_thres = 0.3;
        end

        % final target
        result(To,:) = targetLoc;

        % extend search space in case of failure
        if(target_score<score_thres)
            trans_f = min(1.5, 1.1*trans_f);
            %trans_f = min(3, 1.8*trans_f);

        else
            trans_f = opts.trans_f;
        end

        % bbox regression
        if(opts.bbreg && target_score>0)
            X_ = permute(feat_conv(:,:,:,idx(1:5)),[4,3,1,2]);
            
            X1_ = X_(:,:,7:8,6:9);
            X2_ = X_(:,:,6:6,7:8);
            X3_ = X_(:,:,9:9,7:8);
            X1_ = X1_(:,:);
            X2_ = X2_(:,:);
            X3_ = X3_(:,:);
            X_ = cat(2, X1_, X2_, X3_);
            
%             X1_ = X_(:,:,3,3); X2_ = X_(:,:,3,6); X3_ = X_(:,:,3,9); X4_ = X_(:,:,3,12);
%             X5_ = X_(:,:,6,3); X6_ = X_(:,:,6,6); X7_ = X_(:,:,6,9); X8_ = X_(:,:,6,12);        
%             X9_ = X_(:,:,9,3); X10_ = X_(:,:,9,6); X11_ = X_(:,:,9,9); X12_ = X_(:,:,9,12);
%             X13_ = X_(:,:,12,3); X14_ = X_(:,:,12,6); X15_ = X_(:,:,12,9); X16_ = X_(:,:,12,12);        
% 
%             X1_ = X1_(:,:); X2_ = X2_(:,:); X3_ = X3_(:,:); X4_ = X4_(:,:);
%             X5_ = X5_(:,:); X6_ = X6_(:,:); X7_ = X7_(:,:); X8_ = X8_(:,:);        
%             X9_ = X9_(:,:); X10_ = X10_(:,:); X11_ = X11_(:,:); X12_ = X12_(:,:);
%             X13_ = X13_(:,:); X14_ = X14_(:,:); X15_ = X15_(:,:); X16_ = X16_(:,:);
%             X_ = cat(2, X1_, X2_, X3_, X4_, X5_, X6_, X7_, X8_, X9_, X10_, X11_, X12_, X13_, X14_, X15_, X16_);            
            
            bbox_ = samples(idx(1:5),:);
            pred_boxes = predict_bbox_regressor(bbox_reg.model, X_, bbox_);
            result(To,:) = round(mean(pred_boxes,1));
        end

        %% Prepare training data
        
        if(target_score>score_thres)
            pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_update*2, opts, 0.1, 5);
            r = overlap_ratio(pos_examples,targetLoc);
            pos_examples = pos_examples(r>opts.posThr_update,:);
            pos_examples = pos_examples(randsample(end,min(opts.nPos_update,end)),:);

            neg_examples = gen_samples('uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5);
            r = overlap_ratio(neg_examples,targetLoc);
            neg_examples = neg_examples(r<opts.negThr_update,:);
            neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);

            examples = [pos_examples; neg_examples];
            % to bigger crops
            if(opts.crop_largegt)
                examples = loc2bigloc(examples);
            end            
            
            pos_idx = 1:size(pos_examples,1);
            neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);
          
            feat_conv = rpn2t_features_convX_rpn(feat_extract_net, img, examples, opts);
            total_pos_data{To} = feat_conv(:,:,:,pos_idx);
            total_neg_data{To} = feat_conv(:,:,:,neg_idx);
            total_pos_data{To} = permute(total_pos_data{To}, [1 4 3 2]);
            total_neg_data{To} = permute(total_neg_data{To}, [1 4 3 2]);

            success_frames = [success_frames, To];
            if(numel(success_frames)>opts.nFrames_long)
                total_pos_data{success_frames(end-opts.nFrames_long)} = single([]);
            end
            if(numel(success_frames)>opts.nFrames_short)
                total_neg_data{success_frames(end-opts.nFrames_short)} = single([]);
            end
        else
            total_pos_data{To} = single([]);
            total_neg_data{To} = single([]);
        end

        %% Network update
        if((mod(To,opts.update_interval)==0 || target_score<score_thres) && To~=nFrames)
            if (target_score<score_thres) % short-term update
                %pos_data = cell2mat(total_pos_data(success_frames(max(1,end-opts.nFrames_short+1):end)));
                pos_data = total_pos_data(success_frames(max(1,end-opts.nFrames_short+1):end));
                pos_data = [pos_data{:}];
                pos_data = permute(pos_data, [1 4 3 2]);
            else % long-term update
                %pos_data = cell2mat(total_pos_data(success_frames(max(1,end-opts.nFrames_long+1):end)));
                pos_data = total_pos_data(success_frames(max(1,end-opts.nFrames_long+1):end));
                pos_data = [pos_data{:}];
                pos_data = permute(pos_data, [1 4 3 2]);
            end
                %neg_data = cell2mat(total_neg_data(success_frames(max(1,end-opts.nFrames_short+1):end)));
                neg_data = total_neg_data(success_frames(max(1,end-opts.nFrames_short+1):end));
                neg_data = [neg_data{:}];
                neg_data = permute(neg_data, [1 4 3 2]);

            %fprintf();            
            rpn2t_finetune_hnm_rpn(track_net_solver,pos_data,neg_data,opts,opts.maxiter_update);
        end

        spf = toc(spf);
        fprintf('%f seconds  ',spf);

        %% Display
        if display
            hc = get(gca, 'Children'); delete(hc(1:end-1));
            set(hd,'cdata',img); hold on;

            rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 1);
            if(opts.crop_largegt)
                rectangle('Position', targetLoc_big, 'EdgeColor', [0 1 0], 'Linewidth', 1);
            end
            
            set(gca,'position',[0 0 1 1]);

            text(10,10,num2str(To),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
            fprintf('score = %f\n',target_score);
            %% save
            img_save = getframe(gca);
            imwrite(img_save.cdata, strcat(['output/','save_',videoname,'/'], num2str(To), '.png'));
        
            hold off;
            drawnow;
        end
    end

end

