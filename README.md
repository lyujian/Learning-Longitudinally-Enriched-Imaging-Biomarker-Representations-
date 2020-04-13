# Learning-Longitudinally-Enriched-Imaging-Biomarker-Representations-

## Abstract
A critical challenge to use machine learning to perform longitudinal studies of Alzheimer’s Disease (AD) is the missing medical records during the course of AD development. 
To tackle this problem, in this paper we propose a novel formulation to learn an enriched biomarker representation that can simultaneously capture the information conveyed by both baseline neuroimaging records and that by progressive variations of varied counts of available follow-up records over time. 
While the numbers of the brain scans of the participants vary, the learned biomarker representations are of a fixed length, which enable us to use traditional learning models to study AD developments. 
Moreover, we develop an new objective that maximizes the ratio of the summations of a number of  $\ell_1$-norm distances for improved robustness, which, though, is difficult to solve in general. 
Thus we derive a new efficient non-greedy iterative algorithm and rigorously prove its convergence. 
We have performed extensive experiments on the Alzheimer’s Disease Neuroimaging Initiative (ADNI) cohort. 
A performance gain is achieved to predict ten different cognitive scores when we compare the original representations
against the learned representations with enrichments. 
We further observe that the top selected biomarkers by our posed method are in accordance with the known knowledge in AD studies. 
In addition, we explore the functional brain connectivity from a new perspective using the learned projections
The promising results have demonstrated improved performances of our new method that validate its effectiveness.

## Code

The organization of this repository is as follows:

1. `DATA.m` preprocess the data;
2. `MIL_FS_ADNI.m` and `MIL_VBM_ADNI.m` learn the projection using VBM and FS respectively; 
3. `Performance_FS.m` and `CNN_FS_RAVLT.m` learn a enriched representation compare the enriched  representation by using different methods -- LR, RR, Lasso, SVR and CNN using modality of FS;
4. `Performance_VBM.m` and `CNN_VBM_RAVLT.m` learn a enriched representation compare the enriched  representation by using different methods -- LR, RR, Lasso, SVR and CNN using modality of VBM;
