close all
clear
clc

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
  
 for Mdl_idx =  1 :size(X_RAVLT_W,2)
     for KFold_idx  = 1 : 5 
         CV_Indice = crossvalind('kfold',size(X_RAVLT_W{Mdl_idx},1),KFold_Num );
         Indice_train = find (CV_Indice ~= KFold_idx);
         X_train = X_RAVLT_W{Mdl_idx}(Indice_train,:);
         Y_train = Y_RAVLT(Indice_train,:);
         Indice_test = find (CV_Indice == KFold_idx);
         X_test = X_RAVLT_W{Mdl_idx}(Indice_test,:);
         Y_test = Y_RAVLT(Indice_test,:);
         for DX_idx = 1 : size(Y_RAVLT,2)
         Mdl_SVR {Mdl_idx,KFold_idx,DX_idx} = fitrsvm(X_train,Y_train(:,DX_idx),'Standardize',true,'KernelFunction','gaussian');
         Y_predict_SVR{Mdl_idx,KFold_idx,DX_idx}  = predict(Mdl_SVR{Mdl_idx,KFold_idx,DX_idx} ,X_test);
         RMSE_SVR(Mdl_idx,DX_idx,KFold_idx) = sqrt(mean(Y_test(:,DX_idx)-Y_predict_SVR{Mdl_idx,KFold_idx,DX_idx} ).^2);
         
         Mdl_LR{Mdl_idx,KFold_idx,DX_idx} = fitrlinear(X_train,Y_train(:,DX_idx),'Lambda',0);
         Y_predict_LR{Mdl_idx,KFold_idx,DX_idx} = predict(Mdl_LR{Mdl_idx,KFold_idx,DX_idx},X_test);
         RMSE_LR(Mdl_idx,DX_idx,KFold_idx) = sqrt(mean(Y_test(:,DX_idx)-Y_predict_LR{Mdl_idx,KFold_idx}).^2);
         
         Mdl_RR{Mdl_idx,KFold_idx,DX_idx}  = fitrlinear(X_train,Y_train(:,DX_idx),'Lambda',10.^(-5:5),'Regularization','ridge');
         Y_predict_RR{Mdl_idx,KFold_idx,DX_idx}  = predict(Mdl_RR{Mdl_idx,KFold_idx,DX_idx} ,X_test);
         Y_predict_RR{Mdl_idx,KFold_idx,DX_idx}  = min(Y_predict_RR{Mdl_idx,KFold_idx,DX_idx} ,[],2);
         RMSE_RR(Mdl_idx,DX_idx,KFold_idx) = sqrt(mean(Y_test(:,DX_idx)-Y_predict_RR{Mdl_idx,KFold_idx}).^2);
         
         Mdl_Lasso{Mdl_idx,KFold_idx,DX_idx} = fitrlinear(X_train,Y_train(:,DX_idx),'Lambda',10.^(-5:5),'Regularization','lasso');
         Y_predict_Lasso{Mdl_idx,KFold_idx,DX_idx}= predict(Mdl_Lasso{Mdl_idx,KFold_idx,DX_idx},X_test);
         Y_predict_Lasso{Mdl_idx,KFold_idx,DX_idx} = min(Y_predict_Lasso{Mdl_idx,KFold_idx,DX_idx},[],2);
         RMSE_Lasso(Mdl_idx,DX_idx,KFold_idx) = sqrt(mean(Y_test(:,DX_idx)-Y_predict_Lasso{Mdl_idx,KFold_idx,DX_idx}).^2);  
         end
     end     
 end
    RMSE_SVR_avg_RAVLT = mean(RMSE_SVR,3);
    Std_SVR_avg_RAVLT  = std(RMSE_SVR,0,3);
    RMSE_LR_avg_RAVLT  = mean(RMSE_LR,3);
    Std_LR_avg_RAVLT   = std(RMSE_LR,0,3);
    RMSE_RR_avg_RAVLT  = mean(RMSE_RR,3);
    Std_RR_avg_RAVLT   = std(RMSE_RR,0,3);
    RMSE_Lasso_avg_RAVLT = mean(RMSE_Lasso,3);
    Std_Lasso_avg_RAVLT  = std(RMSE_Lasso,0,3);


    save('R_VBM_RAVLT.mat','RMSE_SVR_avg_RAVLT','RMSE_LR_avg_RAVLT','RMSE_RR_avg_RAVLT','RMSE_Lasso_avg_RAVLT',...
          'Std_SVR_avg_RAVLT','Std_LR_avg_RAVLT','Std_RR_avg_RAVLT','Std_Lasso_avg_RAVLT');