# m5_data_extractor
m5 forecasting data extractor

The dataset comes from the [Kaggle Contest](https://www.kaggle.com/c/m5-forecasting-accuracy).

Remember to cite if you would like to use. Ref paper  "The M5 Accuracy competition: Results, findings and conclusions".

## Propose

The repo is to store the data extractor, which outputs the data InventoryManageEnv used from the raw data.

## Usage

Please see the `data_process.py` for more details in comments at the `config` bar.

The config here is essentially for filter the data and extract data.

Run the script and enter `yes` as input, data and inventory_config.py will be generated.

Essentially, this script looks for items that meet the specified requirements, and extract a certain number (specified) of them to the test dataset.

For the unpicked remainder, they goes to the train dataset.

The directory will be like:

```directory
|root
|inventory_config.py
--|data_patch
----|test
------|sku_selection.txt    (info that tells you what items are present here)
------|store{i}.csv         (testing dataset)
----|train
------|sku_selection.txt
------|store{i}.csv         (noted that training and testing datasets are different)
```