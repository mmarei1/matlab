% Script to run custom training loop
function [trainedNet, outputs] = runCustomTrain(dlnet, options, XTrain_tgt, YTrain_tgt, XTrain_src, YTrain_src)

% set default training options 
if isempty(options)
    options = loadDefaultTrainingOptions("ft",'none');
end

% prepare plotting function
plots = "training-progress";
if plots == "training-progress"
    figure(20);
    plot([0:1:2],[0,0,0],'r');
    hold on;
    plot([0:1:2],[0,0,0],'b');
    lineLossTrain = animatedline([0],[0],'Color','red');
    lineLossValidation = animatedline([0],[0],'Color','blue');
    xlabel("Iteration");
    ylabel("Model Loss");
    legend("Training","Validation");
end

iteration = 0;
start = tic;

numEpochs = options.MaxEpochs;
for epoch = 1:numEpochs
    %shuffle the target training set
    idx_tgt = randperm(numel(YTrain_tgt));
    XTrain_tgt = XTrain_tgt(:,:,:,idx_tgt);
    YTrain_tgt = YTrain_tgt(idx_tgt);
    
    % shuffle the source training set
    idx_src = randperm(numel(YTrain_src));
    XTrain_src = XTrain_src(:,:,:,idx_src);
    YTrain_src = YTrain_src(:,:,:,idx_src);
    
    % loop over mini-batch
    for i = 1:numIterationsPerEpoch
        iteration = iteration+1;
        
        % current index increases by 1
        idx_tgt = (i-1)*miniBatchSize+1:i*miniBatchSize;
        X_tgt = XTrain_tgt(:,:,:,idx_tgt);
        
        % For a regression loss, there are no classes.
        % Instead, create dummy variable for outputs
        % with the same dimension as the output of the final fc layer
        %dlY_tgt = zeros(1, miniBatchSize,'single'); 
        dlY_tgt =  dlarray(single(Y_Train_tgt),'S');
        
        % convert mini-batch of target data into dlarray
        dlX_tgt = dlarray(single(X_tgt),'SSCB');
        
        % convrt mini-batch of source data into dlarray
        dlX_src = dlarray(single(X_src),'SSCB');
        
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX_tgt = gpuArray(dlX_tgt);
            dlX_src = gpuArray(dlX_src);
        end
        % final network layer is a fully connected layer
        % so compute sigmoid as estimates of these values
        
        % Forward-propagate the model and compute its outputs before the
        % sigmoid operation
        % Output of the final fc layer
        %dlZ_presig = forward(dlnet,dlX_tgt);
        %dlZ_sig = sigmoid(dlZ_presig);
        %dlZ_fc = fullyconnect(dlnet, dlZ_sig);
        
        % Prediction outputs without backprop
        dlYPred = predict(dlnet,dlZ_fc);
        
        %Y_est = predict(dlnet,Y_tgt);
        % compute the training NMSE-NMMD loss here
        %[trainingLoss] = compoundLoss(dlnet, dlX_tgt, dlX_src, dlY_tgt);
        
        [gradients_loss, gradients_lp, modelState] = dlfeval(@compoundModelLoss,dlnet,dlYPred,dlX_tgt,dlX_src,dlY_tgt);
        
        learnRate = ilr/(1+decay*iteration);
        
        [dlnet.Learnables] = adamupdate(dlnet.Learnables, gradients, learnRate, momentum);
        
        % Add the new training loss calculated to the training loss line every
        % validation interval
        if mod(iteration,validationInterval) ~= 0
            info.TrainingLoss(iteration) = double(gather(extractdata(trainingLoss)));
            figure(20);
             D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        elseif mod(iteration,validationInterval) == 0
            info.TrainingLoss(iteration) = double(gather(extractdata(trainingLoss)));
            info.ValidationLoss(iteration) = double(gather(extractdata(validationLoss)));
            figure(20);
            addpoints(lineLossValidation,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
        
    end
    trainedNet = dlnet;
    end
end