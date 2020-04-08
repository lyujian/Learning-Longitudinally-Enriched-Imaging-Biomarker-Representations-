clear
clc

load('ADNI_VBM_MIL.mat')

for i = 1 : size(ADNI_VBM_MIL,2)-1
    VBM_MIL{i}.VBM = ADNI_VBM_MIL{i}.Base;
    VBM_MIL{i}.Instance = ADNI_VBM_MIL{i}.Instance;
    VBM_MIL{i}.DX_BL = ADNI_VBM_MIL{i}.DX_BL;
    VBM_MIL{i}.ADAS_BL = ADNI_VBM_MIL{i}.ADAS_BL;
    VBM_MIL{i}.FLU_BL = ADNI_VBM_MIL{i}.FLU_BL;
    VBM_MIL{i}.MMSE_BL = ADNI_VBM_MIL{i}.MMSE_BL;
    VBM_MIL{i}.RAVLT_BL = ADNI_VBM_MIL{i}.RAVLT_BL;
    VBM_MIL{i}.TRAILS_BL = ADNI_VBM_MIL{i}.TRAILS_BL;    
end

save('VBM_MIL.mat','VBM_MIL')