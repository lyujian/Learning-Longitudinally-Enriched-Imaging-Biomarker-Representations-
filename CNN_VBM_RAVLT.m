close all
clear


load('ADNI_VBM_MIL.mat')

for W_idx = 1 : size(ADNI_VBM_MIL{1}.Weight,2)
    X_RAVLT=[];
    for idx = 1 : size(ADNI_VBM_MIL,2)-1
        if sum(isnan(ADNI_VBM_MIL{idx}.RAVLT_BL))==0 && sum(isempty(ADNI_VBM_MIL{idx}.RAVLT_BL))==0 && sum(sum(isnan(ADNI_VBM_MIL{idx}.Weight{W_idx})))==0
            X_RAVLT = [X_RAVLT;ADNI_VBM_MIL{idx}.Base*ADNI_VBM_MIL{idx}.Weight{W_idx}'];
        end
    end
    X_RAVLT_W{W_idx}=X_RAVLT;
end
X_RAVLT=[];Y_RAVLT=[];
for idx = 1 : size(ADNI_VBM_MIL,2)-1
    if sum(isnan(ADNI_VBM_MIL{idx}.RAVLT_BL))==0 && sum(isempty(ADNI_VBM_MIL{idx}.RAVLT_BL))==0
        X_RAVLT = [X_RAVLT;ADNI_VBM_MIL{idx}.Base];
        Y_RAVLT = [Y_RAVLT;ADNI_VBM_MIL{idx}.RAVLT_BL];
    end
end
X_RAVLT_W{W_idx+1}=X_RAVLT;
KFold_Num = 5;





for Mdl_idx =  4 :size(X_RAVLT_W,2)
    for KFold_idx  = 1 : KFold_Num
        CV_Indice = crossvalind('kfold',size(X_RAVLT_W{Mdl_idx},1),KFold_Num );
        Indice_train = find (CV_Indice ~= KFold_idx);
        X_train = X_RAVLT_W{Mdl_idx}(Indice_train,:);
        Y_train = Y_RAVLT(Indice_train,:);
        Indice_test = find (CV_Indice == KFold_idx);
        X_test = X_RAVLT_W{Mdl_idx}(Indice_test,:);
        Y_test = Y_RAVLT(Indice_test,:);
        for DX_idx = 1 : size(Y_train,2)
            
            CNN_Training_data = reshape(X_train',[1,size(X_train,2),1,size(X_train,1)]);
            CNN_Label_data = reshape(Y_train(:,DX_idx)',[1,1,1,size(Y_train,1)] );
            CNN_Test_data = reshape(X_test',[1,size(X_test,2),1,size(X_test,1)]);
            
            layers = [
                imageInputLayer([1 size(CNN_Training_data,2) 1])                
                convolution2dLayer([1 5], 16)
                reluLayer
                convolution2dLayer([1 10], 32)
                reluLayer
                maxPooling2dLayer([1 2])            
                dropoutLayer(0.3)
                fullyConnectedLayer(1)
                regressionLayer];
            
            options = trainingOptions('sgdm', ...
                'MiniBatchSize',16, ...
                'MaxEpochs',30, ...
                'InitialLearnRate',1e-4, ...
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropFactor',0.2, ...
                'LearnRateDropPeriod',10, ...
                'Verbose',1);
            net = trainNetwork(CNN_Training_data,CNN_Label_data,layers,options);
            YPredicted = predict(net,CNN_Test_data);
            RMSE_CNN(Mdl_idx-3,DX_idx,KFold_idx) = sqrt(mean(Y_test(:,DX_idx)-YPredicted).^2);           
        end
    end
end
 
 
 
 RMSE_CNN_RAVLT_avg = mean(RMSE_CNN,3);
 Std_CNN_RAVLT      = std(RMSE_CNN,0,3);
 save('CNN_VBM_RAVLT.mat','RMSE_CNN_RAVLT_avg','Std_CNN_RAVLT');
 