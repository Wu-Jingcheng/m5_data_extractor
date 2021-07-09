template = \
'''
# Configuration ===============================================================================

# 配置仿真环境

# How to simulate demand, ['GAMMA', 'DYNAMIC_GAMMA', 'ONLINE']
# "GAMMA": generate demands according to predefined paramter, store them and apply them to all schedulers
# "DYNAMIC_GAMMA": Generate demands on-the-fly for all schedulers, which means different rounds may see different demands;
# "ONLINE": Read demand from files whose distribution is unknown.
# demand_sampler = 'DYNAMIC_GAMMA'
demand_sampler = 'ONLINE'

#SKU CONFIG
sku_config = {{
    "sku_num": {sku_num},
    "sku_names": ['SKU%i' % i for i in range(1, {sku_num}+1)]
}}

# SUPPLIER_CONFIG
supplier_config = {{
    "name": 'SUPPLIER',
    "short_name": "M",
    "supplier_num": 1,
    "fleet_size": [500],
    "unit_storage_cost": [1],
    "unit_transport_cost": [2],
    "storage_capacity": [30000],
    "order_cost": [200],
    "delay_order_penalty": [1000],
    "downstream_facilities": [[0]], 
    "sku_relation": [
        [{supp_rela}
        ]
    ]
}}

# WAREHOUSE_CONFIG
# The length of warehouse_config corresponds to number of intermedia echelons
warehouse_config = [
    {{
        "name": "WAREHOUSE",
        "short_name": "R",
        "warehouse_num": 1,
        "fleet_size": [500],
        "unit_storage_cost": [1],
        "unit_transport_cost": [1],
        "storage_capacity": [2000000],
        "order_cost": [200],
        "delay_order_penalty": [1000], 
        "downstream_facilities": [[0]],
        "sku_relation": [
            [{w1_rela}
            ]
        ]
    }},
    {{
        "name": "WAREHOUSE",
        "short_name": "F",
        "warehouse_num": 1,
        "fleet_size": [500],
        "unit_storage_cost": [1],
        "unit_transport_cost": [1],
        "storage_capacity": [100000],
        "order_cost": [200],
        "delay_order_penalty": [1000], 
        "downstream_facilities": [[0]],
        "sku_relation": [
            [{w2_rela}
            ]
        ]
    }}
]

# store_CONFIG
store_config = {{
    "name": "STORE",
    "short_name": "S",
    "store_num": 1,
    "storage_capacity": [1000],
    "unit_storage_cost": [5],
    "order_cost": [100],
    "sku_relation": [
        [{store_rela}
        ]
    ]
}}


# CONSUMER_NETWORK_CONFIG

env_config = {{
    'global_reward_weight_producer': 0.50,
    'global_reward_weight_consumer': 0.50,
    'downsampling_rate': 1,
    'episod_duration': 365*4, #365*4
    'evaluation_len': 365, #365
    "initial_balance": 100000,
    "consumption_hist_len": 21,
    "sale_hist_len": 21,
    "demand_prediction_len": 21,
    "uncontrollable_part_state_len": 21*7, #31
    "uncontrollable_part_pred_len": 6, # 3天具体，3天均值，7天均值，21天均值
    "total_echelons": 4,
    "init": None, # or `rnd`, None 'rst'
    "tail_timesteps": 0,
    "training": True
}}
'''