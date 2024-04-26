#!/bin/bash

# Eighth demand
python run_simopt.py +experiment=rs_eight_product_perishable/eighth_demand/exact_match_max_sub_shortage_alt_rula
python run_simopt.py +experiment=rs_eight_product_perishable/eighth_demand/exact_match_max_sub_wastage_alt_rula

python run_simopt.py +experiment=rs_eight_product_perishable/eighth_demand/oldest_compatible_match_max_sub_shortage_alt_rula
python run_simopt.py +experiment=rs_eight_product_perishable/eighth_demand/oldest_compatible_match_max_sub_wastage_alt_rula

python run_simopt.py +experiment=rs_eight_product_perishable/eighth_demand/priority_match_max_sub_shortage_alt_rula
python run_simopt.py +experiment=rs_eight_product_perishable/eighth_demand/priority_match_max_sub_wastage_alt_rula

# Full demand
python run_simopt.py +experiment=rs_eight_product_perishable/full_demand/exact_match_max_sub_shortage_alt_rula
python run_simopt.py +experiment=rs_eight_product_perishable/full_demand/exact_match_max_sub_wastage_alt_rula

python run_simopt.py +experiment=rs_eight_product_perishable/full_demand/oldest_compatible_match_max_sub_shortage_alt_rula
python run_simopt.py +experiment=rs_eight_product_perishable/full_demand/oldest_compatible_match_max_sub_wastage_alt_rula

python run_simopt.py +experiment=rs_eight_product_perishable/full_demand/priority_match_max_sub_shortage_alt_rula
python run_simopt.py +experiment=rs_eight_product_perishable/full_demand/priority_match_max_sub_wastage_alt_rula

