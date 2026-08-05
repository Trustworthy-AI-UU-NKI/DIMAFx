# Instructions on running DIMAFx
## 1. Data preprocessing
First , prepare the data such that DIMAFx can use it. Please see the [README](data/README.md) in the `data` folder for detailed instructions on how to download, preprocesss and structure the data. It also contains instructions on how to create the data for the censoring sensitivity analysis (Figure E.1 in the paper)
Currently, it supports the 4 [TCGA](https://portal.gdc.cancer.gov) data cohorts used in the paper, i.e., BRCA, BLCA, LUAD and KIRC. However, it can easily be adapted to other cohorts also. After this step, cd back to the current directory (`src`).


## 2. Constructing initial histology prototypes
DIMAFx obtains initial prototypes as means for the mixture distributions by clustering the train data and taking the cluster centres as initial means. To construct these initial mixture distribution means (prototypes) for the TCGA-BRCA dataset, run 
```
python main_prototype.py --data_source data/data_files/tcga_brca \
                         --wsi_dir wsi/extracted_res0_5_patch256_uni/feats_h5/ \
                         --in_dim 1024 \
                         --mode faiss \
                         --n_proto 16 
```
**List of arguments:**
- `data_source`: Path to the dataset.
- `wsi_dir`: Subpath to extracted WSI features.
- `in_dim`: Dimension of the patch representations obtained from the foundation model.
- `mode`: Clustering method (faiss for GPU, kmeans for CPU, default = faiss).
- `n_proto`: Number of prototypes (default = 16).

For all possible arguments, see `main_prototype.py`. These default values reproduce the DIMAFx and DIMAF settings used in our paper. To run the ablation variants with a different number of prototypes (N_h), only change `n_proto`. For the other ablations and censoring experiments, we use the same settings here as DIMAFx.


## 3. Train and test for survival prediction
To train DIMAFx on the TCGA-BRCA dataset, run
```
python main_survival.py --data_source data/data_files/tcga_brca/ \
                        --max_epochs 30 \
                        --task dss_survival_brca \
                        --exp_code DIMAFx \
                        --loss_fn cox_distcor \
                        --omics_type rna_data \
                        --w_dis 7 \
                        --w_surv 1 \
                        --mode train_test \
                        --aggr_post_embed weighted_mean \
                        --wsi_repr importance

```
**List of arguments:**
- `data_source`: Path of the data, default is _data/data_files/tcga_brca/. Adjust to use other cohorts by replacing `brca` with `blca`, `kirc` or `luad`.
- `max_epochs`: Number of epochs the model will train for, default is 30.
- `task`: Defines the task of the model. Options are [`dss_survival_brca`, `dss_survival_blca`, `dss_survival_luad`, `dss_survival_kirc`].
- `exp_code`: Name of the experiment.
- `loss_fn`: Loss function used, default is cox_distcor (Cox + DC loss).
- `omics_type`: Name of the rna data file.
- `w_dis`: Weight of the disentanglement loss in the total loss, default is 7.
- `w_surv`: Weight of the survival loss in the total loss, default is 1.
- `mode`: Determines the mode of the experiment. Options are train, test, train_test and shap.
- `aggr_post_embed`: Determines how the features in the disentangled representations are aggregated after Disentangled Attention Fusion.
- `wsi_repr`: Determines which WSI representation is used. DIMAFx uses `importance`. 


For all possible arguments, see `main_survival.py`. These default values reproduce the DIMAFx settings used in our paper. 

### Ablations
If otherwise mentioned, use the same arguments as above. Don't forget to change `exp_code` to the name of the specific run/experiment!
- If you want to run DIMAF, put: `aggr_post_embed` to `mean` and `wsi_repr` to `normal` 
- If you want to run the ablation version without the disentanglement loss, put: `loss_fn` to `cox`
- If you want to run DIMAFx_pre_card: put `wsi_repr` to `normal`
- If you want to run DIMAFx_mean_aggr: put `aggr_post_embed` to `mean`
- If you want to run the ablations with the different weights for the disentanglement loss: put `w_dis` to the corresponding weight (default=7).
- If you want to run the ablations with the different number of WSI prototypes: put `n_proto` to the number of prototypes and change `proto_file` to the corresponding prototype file you created in Section 2. 

To evaluate survival and disentanglement performance across folds (mean ± std), use `get_results.ipynb`.

## 4. Kaplan-Meier analysis
When you have run all your experiments, run the following command to get the Kaplan-Meier survival curves for DIMAFx.
```
python plot_KM_curves.py --exp_code DIMAFx
```

If you want to obtain the KM curves for DIMAF, change `--exp_code` to `DIMAF`


## 5. Running SHAP for feature importance of the disentangled representations
To compute the SHAP values, run the same command as training but set `--mode shap` and include `--shap_refdist_n` and `--explainer`:
- `shap_refdist_n`: Size of the background samples (training set). Recommended:
    - BRCA: 512
    - BLCA: 256
    - LUAD: 320
    - KIRC: 192
- `explainer`: Explanation technique (`shap` by default, also supports `eg` for Expected Gradients).

To check whether the SHAP values and rankings are stable, two settings can be varied:

- **Background set size** — recompute the SHAP values with a different `--shap_refdist_n`. The size must be a multiple of 64 and no larger than the original training set.
- **Background seed** — set `--mode shap_stable` and include `--shap_refdist_n` and `--explainer` as needed. This will compute the SHAP values for 5 different seeds and save them.

The code for computing correlations between the SHAP values obtained under these different settings is in [this notebook](interpretability/shap_background_stability.ipynb).


## 6. Interpretability analysis
For the interpretability analysis of DIMAFx, see the [README](interpretability/README.md) in  the `interpretability` folder.