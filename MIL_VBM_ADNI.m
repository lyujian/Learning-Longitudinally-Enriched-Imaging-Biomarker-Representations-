clear
close all

load('ADNI_info.mat')
MIL_idx = 1;
for idx = 1 : size(ADNI_info,2)
    if sum(isnan(ADNI_info{idx}.VBM_BL))==0 % &&  sum(isnan(ADNI_info{idx}.ADAS_BL))==0
        % ADNI_VBM_MIL{MIL_idx}.Label     = ADNI_info{idx}.ADAS_BL;
        ADNI_VBM_MIL{MIL_idx}.Base      = ADNI_info{idx}.VBM_BL;
        ADNI_VBM_MIL{MIL_idx}.Instance  = [];
        ADNI_VBM_MIL{MIL_idx}.ADAS_BL   = ADNI_info{idx}.ADAS_BL;
        ADNI_VBM_MIL{MIL_idx}.DX_BL     = ADNI_info{idx}.DX_BL;
        ADNI_VBM_MIL{MIL_idx}.FLU_BL    = ADNI_info{idx}.FLU_BL;
        ADNI_VBM_MIL{MIL_idx}.MMSE_BL   = ADNI_info{idx}.MMSE_BL;
        ADNI_VBM_MIL{MIL_idx}.RAVLT_BL  = ADNI_info{idx}.RAVLT_BL;
        ADNI_VBM_MIL{MIL_idx}.TRAILS_BL = ADNI_info{idx}.TRAILS_BL;
                            
        if sum(isnan(ADNI_info{idx}.VBM_M6))==0 &&  sum(isempty(ADNI_info{idx}.VBM_M6))==0
            ADNI_VBM_MIL{MIL_idx}.Instance  = [ADNI_VBM_MIL{MIL_idx}.Instance;ADNI_info{idx}.VBM_M6];
        end
        
        if sum(isnan(ADNI_info{idx}.VBM_M12))==0 &&  sum(isempty(ADNI_info{idx}.VBM_M12))==0
            ADNI_VBM_MIL{MIL_idx}.Instance  = [ADNI_VBM_MIL{MIL_idx}.Instance;ADNI_info{idx}.VBM_M12];
        end
        
        if sum(isnan(ADNI_info{idx}.VBM_M18))==0 &&  sum(isempty(ADNI_info{idx}.VBM_M18))==0
            ADNI_VBM_MIL{MIL_idx}.Instance  = [ADNI_VBM_MIL{MIL_idx}.Instance;ADNI_info{idx}.VBM_M18];
        end
        
        if sum(isnan(ADNI_info{idx}.VBM_M24))==0 &&  sum(isempty(ADNI_info{idx}.VBM_M24))==0
            ADNI_VBM_MIL{MIL_idx}.Instance  = [ADNI_VBM_MIL{MIL_idx}.Instance;ADNI_info{idx}.VBM_M24];
        end
        
        if sum(isnan(ADNI_info{idx}.VBM_M36))==0 &&  sum(isempty(ADNI_info{idx}.VBM_M36))==0
            ADNI_VBM_MIL{MIL_idx}.Instance  = [ADNI_VBM_MIL{MIL_idx}.Instance;ADNI_info{idx}.VBM_M36];
        end
        
        if sum(isempty(ADNI_VBM_MIL{MIL_idx}.Instance))==0 && size(ADNI_VBM_MIL{MIL_idx}.Instance,1)>2
            MIL_idx = MIL_idx + 1;
        end
        
    end
end


% if isempty( ADNI_VBM_MIL{size(ADNI_VBM_MIL,2)}.Instance)
%      ADNI_VBM_MIL =  ADNI_VBM_MIL{:,1:size(ADNI_VBM_MIL,2)-1};
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dim = size(ADNI_VBM_MIL{1}.Base,2);

knn_num = 2;

for idx =  1 : size(ADNI_VBM_MIL,2)-1
    for Dim_r = 10 : 10 : Dim
        W = randn (Dim_r,Dim);
        W = (W*W')^(-1/2)*W;
        X =  ADNI_VBM_MIL{idx}.Instance;
        X_num = size(X,1);
        X_mean = mean(X);
        
        %caculate B
        B=[];
        for i = 1 : X_num
            x_i = X(i,:);
            Idx = knnsearch(X,x_i,'K',2,'Distance','euclidean');
            B   = [B;mean(X(Idx,:))-mean(X)];
        end
       
        %caculate Lambda
        Lambda=[];
%         step = 2;
%         Lambda(1) = eps;
%         Lambda(2) = 2*eps;  
%         while Lambda(step)/Lambda(step-1) >= 1.00001        
%         step = step + 1;
        for step = 1 : 200
            r1 = 0;
            r2 = - eps;
            A_g =zeros(Dim_r,knn_num);
            Idx = {};
            for i = 1 : X_num
                x_i = X(i,:);
                r1 = r1 + sum(abs(W*(x_i-X_mean)'));
                Idx{i} =  knnsearch(X,x_i,'K',knn_num,'Distance','euclidean');
                r2 = r2 + sum(sum(abs(W*(X(Idx{i},:)-mean(X(Idx{i},:)))')));
                for idx_w = 1 :Dim_r
                    for idx_knn = 1 : knn_num
                        A_g_t = X(Idx{i}(idx_knn),:)-mean(X(Idx{i},:));
                        A_g(idx_w,idx_knn) = (A_g_t*A_g_t')/abs(W(idx_w,:)*A_g_t'+eps);
                    end
                end
                A_g = sum(A_g,2);
            end
            Lambda(step) = r1 /r2;

            % Find W
            F_W_1 = eps;F_W_2 = 2*eps;
            beta=0.9;
            m=1;
            while F_W_2 > F_W_1
                L_1 = B' * ((B*W')./abs(B*W'));
                F_W_1 = F_W_2;
                L_2 = A_g.*W;
                G_w = L_1'-Lambda(step)*L_2;
                W_n = W + beta^m*G_w;
                W   = abs((W_n*W_n')^(-1/2)*W_n);
                
                H_W=0;M_W=0;
                for i = 1
                    H_W = H_W + sum(abs(W*(x_i-X_mean)'));
                    M_W = M_W + sum(sum(abs(W*(X(Idx{i},:)-mean(X(Idx{i},:)))')));
                end
                F_W_2 = H_W -Lambda(step)*H_W;
                m = m +1;
            end
            
        end
        
        ADNI_VBM_MIL{idx}.Weight{Dim_r/10} = W;
        ADNI_VBM_MIL{idx}.Lambda{Dim_r/10}= Lambda;
    end
end
save('ADNI_VBM_MIL.mat','ADNI_VBM_MIL');

