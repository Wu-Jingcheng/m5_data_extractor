'''
* id: HOBBIES_1_001_CA_1_validation
* item_id: HOBBIES_1_001
* dept_id: HOBBIES_1 (FOODS_1~3, HOBBIES?, HOUSEHOLD?)
* cat_id: cat_id
* store_id: CA_1
* d_1~d_1941 (train) d_1~d_1913(val)
'''
import pandas as pd
import numpy as np
import os
from random import shuffle, seed, randint
from itertools import combinations
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()
seed(args.seed)

# config here
sales_mean_threshold = 0.5
max_zero_cnt = 500
max_nosale_cnt = 15
sku_num = 50
# stores = None
stores = ['CA_3']
store_num = len(stores)
avoid_items = [
    # "FOODS_3_080",
    # "FOODS_3_228",
    # "HOBBIES_1_004",
    # "HOBBIES_1_323",
    # "HOUSEHOLD_1_234",
    # "HOUSEHOLD_1_434",
]
time_span = 1941
cats_all = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
# prices = [200, 300, 400, 500, 220, 330, 440, 550, 540, 250, 350, 450]
# costs = [100, 180, 270, 390, 110, 220, 330, 440, 400, 120, 230, 300]
costs = [randint(100, 300)//10*10 for _ in range(sku_num)]
prices = [costs[i]+randint(100, 150)//10*10 for i in range(sku_num)]
# config here


date_start_idx = 6

def select(df, sales_mean):
    zeros_cnt = (df.iloc[:, date_start_idx:]==0).astype(int).sum(axis=1)
    a = df.iloc[:, date_start_idx:].astype(int).values
    max_consecutive_zeros = []
    for row in a:
        m = np.r_[False, row==0, False]
        idx = np.flatnonzero(m[:-1]!=m[1:])
        out = (idx[1::2]-idx[::2]).max() if len(idx)>0 else 0
        max_consecutive_zeros.append(out)
    max_consecutive_zeros = np.array(max_consecutive_zeros)
    # print(sales_mean)
    # print(zeros_cnt)
    return (sales_mean>sales_mean_threshold) & (zeros_cnt<max_zero_cnt) &\
         (max_consecutive_zeros< max_nosale_cnt)

def check(item_stores):
    return (len(item_stores)>=store_num and stores==None) or all([(store in item_stores) for store in stores])

def parse_chart():
    chart_path = "sales_train_evaluation.csv"

    df = pd.read_csv(chart_path)
    sales_mean = df.iloc[:, date_start_idx:].mean(axis=1)

    selected = select(df, sales_mean)
    fdf = df[selected]
    fsm = sales_mean[selected]

    items = {}
    cats = {}
    sms = {}
    for idx in range(fdf.shape[0]):
        item = fdf['item_id'].iloc[idx]
        cat = fdf['cat_id'].iloc[idx]
        store = fdf['store_id'].iloc[idx]
        smean = fsm.iloc[idx]
        sales = fdf.iloc[idx, date_start_idx:]
        if item not in items:
            items[item] = {}
            # cats[item] = {}
            sms[item] = {}
        items[item][store] = sales
        cats[item] = cat
        sms[item][store] = smean

    item_cnt = 0
    items_todel = []
    for item in items.keys():
        if check(items[item]):
            if item in avoid_items:
                continue
            item_cnt += 1
            # print(f"{item}: {list(zip(items[item].keys(), sms[item].values(), [cats[item]]*len(items[item])))}")
        else:
            items_todel.append(item)
    
    for item in items_todel:
        del items[item]
        del cats[item]
        del sms[item]
    print(f"{item_cnt} items meets our requirements")
    return items, cats, sms

def generate_charts_and_dscrp(chosen_items, items, cats, sms, dir_name, sku_num=sku_num):
    dates = pd.date_range("2011-1-29", periods=time_span, freq="D")
    dates = dates.strftime("%m/%d/%Y")
    cols = ["SKU", "DT", "Sales", "Prices", "SaleMean"]
    base_dir = 'data_patch/'
    try:
        os.mkdir(base_dir)
    except OSError:
        pass
    base_dir = os.path.join(base_dir, dir_name)
    try:
        os.mkdir(base_dir)
    except OSError:
        pass

    dfs = [pd.DataFrame(index=range(sku_num*time_span), columns=cols) for _ in range(store_num)]
    sku_selection = ''
    for i, df in enumerate(dfs):
        store_key = stores[i]
        for sku_idx in range(1, sku_num+1):
            item_key = chosen_items[sku_idx-1]
            sku_selection += f'Store{i+1} SKU{sku_idx} is {item_key} at {store_key}\n'

            sales = items[item_key][store_key]
            salemean = sms[item_key][store_key]

            df['SKU'].iloc[(sku_idx-1)*time_span:sku_idx*time_span] = f"SKU{sku_idx}"
            df['DT'].iloc[(sku_idx-1)*time_span:sku_idx*time_span] = dates
            df['Sales'].iloc[(sku_idx-1)*time_span:sku_idx*time_span] = np.array(sales)
            df['Prices'].iloc[(sku_idx-1)*time_span:sku_idx*time_span] = prices[(sku_idx-1)%len(prices)]
            df['SaleMean'].iloc[(sku_idx-1)*time_span:sku_idx*time_span] = salemean
        df.to_csv(os.path.join(base_dir, f'store{i+1}.csv'), index=False, date_format = "%s")
    with open(os.path.join(base_dir, 'sku_selection.txt'), 'w') as f:
        f.write(sku_selection)

def make_data(items, cats, sms):
    import numpy as np
    item_keys = list(items.keys())
    shuffle(item_keys)
    test_items = item_keys[:sku_num]
    train_items = item_keys[sku_num:]
    train_size = len(train_items)
    print(f'test_items: {len(test_items)}')
    print(f'train_items: {train_size}')
    res = input("Ready to make? type [yes].")
    if res=='yes':
        generate_charts_and_dscrp(test_items, items, cats, sms, 'test')
        generate_charts_and_dscrp(train_items, items, cats, sms, 'train', sku_num=train_size)
    # num = np.math.factorial(train_size)//np.math.factorial(sku_num)//np.math.factorial(train_size-sku_num)
    # ans = input(f"There are {num} combinations, sure to proceed?(enter [yes])")
    # if ans=='yes':
    #     train_combs = combinations(train_items, sku_num)
    #     for idx, train_subset in enumerate(train_combs):
    #         generate_charts_and_dscrp(train_subset, items, cats, sms, f"train{idx+1}")

    # average_num = sku_num //len(cats_all)
    # item_nums = [average_num]*len(cats_all)
    # item_cnts = [0]*len(cats_all)
    # item_nums[-1]+=sku_num%len(cats_all)

    # chosen_items = []
    # for i, num in enumerate(item_nums):
    #     cnt = 0
    #     cat = cats_all[i]
    #     for key in item_keys:
    #         if cats[key]==cat:
    #             cnt += 1
    #             if cnt<=item_nums[i]:
    #                 chosen_items.append(key)
    #     item_cnts[i] = cnt
    # print("item_cats", cats_all)
    # print("item_cnts", item_cnts)
    # print("item_nums", item_nums)
    # assert len(chosen_items)==sku_num

def make_config():
    sku_names = ['SKU%i'%i for i in range(1, sku_num+1)]
    vlts = [randint(1,3) for _ in range(sku_num)]
    init_stocks = [randint(2,10)*10 for _ in range(sku_num)]
    supp_rela = '\n             '.join(['{'+\
        '''"sku_name": "{sku_name}", "price": {cost}, "cost": {cost_minus_50}, "service_level": 0.95, "vlt": {vlt}, "penalty": 1000, 'init_stock': 1000, "production_rate": 200'''.format(
            cost = costs[i], vlt = vlts[i], sku_name = sku_names[i], cost_minus_50 = costs[i]-50
        )\
        +'},'for i in range(sku_num)])
    w1_rela = '\n             '.join(['{'+\
        '''"sku_name": "{sku_name}", "price": {cost}, "cost": {cost}, "service_level": 0.95, "vlt": {vlt}, 'init_stock': 100000'''.format(
            cost = costs[i], vlt = vlts[i], sku_name = sku_names[i], price = prices[i]
        )\
    +'},'for i in range(sku_num)])
    w2_rela = '\n             '.join(['{'+\
        '''"sku_name": "{sku_name}", "price": {cost}, "cost": {cost}, "service_level": 0.95, "vlt": {vlt}, 'init_stock': 10000'''.format(
            cost = costs[i], vlt = vlts[i], sku_name = sku_names[i], price = prices[i]
        )\
    +'},'for i in range(sku_num)])
    store_rela = '\n             '.join(['{'+\
        '''"sku_name": "{sku_name}", "price": {price}, "service_level": 0.95, "cost": {cost}, "sale_gamma": 50, 'init_stock': {init_stock}, 'max_stock': 400'''.format(
            cost = costs[i], vlt = vlts[i], sku_name = sku_names[i], price = prices[i], init_stock = init_stocks[i]
        )\
    +'},'for i in range(sku_num)])
    # config of config
    config = {
        'supp_rela': supp_rela,
        'w1_rela': w1_rela,
        'w2_rela': w2_rela,
        'store_rela': store_rela,
        'sku_num': sku_num
    }
    from inventory_config_template import template
    res = template.format(**config)
    print(res)
    with open('inventory_config.py', 'w', encoding='utf-8') as f:
        f.write(res)


items, cats, sms = parse_chart()

make_data(items, cats, sms)
make_config()