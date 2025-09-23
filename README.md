# NoMoCLIP
Interpretable Modeling of RNA–Protein Interactions from eCLIP‑Seq Profiles for Motif‑Free RBPs
# ![image](https://github.com/yangyn533/NoMoCLIP/blob/main/NoMoCLIP.png)
## :one: Data availability
[NoMoCLIP_dataset](https://doi.org/10.6084/m9.figshare.26082916.v1)

## :two: Environment Setup
#### 2.1 Create and activate a new virtual environment
```
conda create -n NoMoCLIP python=3.7.16 
conda activate NoMoCLIP
```
#### 2.2 Install the package and other requirements
```
git clone https://github.com/yangyn533/NoMoCLIP
cd NoMoCLIP
python3 -m pip install --editable .
python3 -m pip install -r requirements.txt
```
## :three: Process data

#### 3.1 Sequential encoding
```
python position_inf.py  --set_path <PATH_TO_YOUR_DATA>  --out_path <PATH_TO_YOUR_OUTPUT_DIRECTORY>
```

#### 3.2 Structural encoding
This feature requires the **RNAplfold** tool, which is executed in a **Python 2.7 environment**.
```
python structure_inf.py  --set_path <PATH_TO_YOUR_DATA>  --out_path <PATH_TO_YOUR_OUTPUT_DIRECTORY>
```

#### 3.3 Semantic encoding
```
python attention_graph.py \
  --kmer 1
  --set_path <PATH_TO_YOUR_DATA> \
  --out_path <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
  --model_type <PATH_TO_YOUR_NLP_MODEL> \ 
  --maxlen 101 \
  --device cuda:1 \
  --device1 cuda:1 \
  --device2 cuda:1 
```
#### 3.4 Functional properties

For this feature, you need to use the [corain](https://github.com/idrblab/corain?tab=readme-ov-file#requirements-and-installment). 

```
python instinct_inf.py \
  --base_path <PATH_TO_YOUR_DATA> \
  --set_path <PATH_TO_YOUR_INTERMEDIATE_OUTPUT_DIRECTORY> \
  --out_path <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
  --method_path <PATH_TO_YOUR_CORAIN_DIRECTORY> \ 
  --num 2
```
**Note:** The argument `--num` should be tested with all values in `[2, 3, 5, 7, 10]`.

## :four: Training Process
```
python model_train.py \
  --base_path <PATH_TO_YOUR_DATA_DIRECTORY> \
  --set_path <PATH_TO_YOUR_FEATURE_DIRECTORY> \
  --out_path <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
  --fold 5
```

## :five: Prediction
```
python model_predict.py \
  --set_path <PATH_TO_YOUR_FEATURE_DIRECTORY> \
  --out_path <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
  --model_path <PATH_TO_YOUR_MODEL> \
  --gpu_id 1
```
## 🧬 Motif analysis

Motif extraction requires the installation of the **[MEME Suite](https://meme-suite.org/meme/doc/download.html)** package.

#### 6.1 Sequential motifs

```
python seq_motifs.py \
  --layer <THE_LAYER_OF_MODEL_YOU_SELECTED> \
  --set_path <PATH_TO_YOUR_FEATURE_DIRECTORY> \
  --out_path <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
  --model_path <PATH_TO_YOUR_MODEL> \
  --pwm_path <PATH_TO_YOUR_PWM_FILE> \
  --motif_size 7
  --gpu_id 1
```

#### 6.2 Structural motifs

```
python structure_motifs.py \
  --layer <THE_LAYER_OF_MODEL_YOU_SELECTED> \
  --set_path <PATH_TO_YOUR_FEATURE_DIRECTORY> \
  --out_path <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
  --model_path <PATH_TO_YOUR_MODEL> \
  --motif_size 7
  --gpu_id 1
```

## 📊 High attention regions

```
python high_attention_region.py \
  --set_path <PATH_TO_YOUR_FEATURE_DIRECTORY> \
  --out_path <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
  --model_path <PATH_TO_YOUR_MODEL> \
  --gpu_id 1
```
## Contact
Thank you and enjoy the tool! If you have any suggestions or questions, please email me at [*yangyn533@nenu.edu.cn*](mailto:yangyn533@nenu.edu.cn)*.*
