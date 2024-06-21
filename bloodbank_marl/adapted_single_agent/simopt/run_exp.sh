#!/bin/bash

python run_simopt.py +experiment=eight_product_perishable/base_case_exact_match
python run_simopt.py +experiment=eight_product_perishable/base_case_oldest_compatible_match
python run_simopt.py +experiment=eight_product_perishable/base_case_priority_match

python run_simopt.py +experiment=eight_product_perishable/base_case_exact_match environment.env_params.max_substitution_cost=650
python run_simopt.py +experiment=eight_product_perishable/base_case_oldest_compatible_match environment.env_params.max_substitution_cost=650
python run_simopt.py +experiment=eight_product_perishable/base_case_priority_match environment.env_params.max_substitution_cost=650

python run_simopt.py +experiment=eight_product_perishable/eighth_demand_exact_match
python run_simopt.py +experiment=eight_product_perishable/eighth_demand_oldest_compatible_match
python run_simopt.py +experiment=eight_product_perishable/eighth_demand_priority_match

python run_simopt.py +experiment=eight_product_perishable/eighth_demand_exact_match environment.env_params.max_substitution_cost=650
python run_simopt.py +experiment=eight_product_perishable/eighth_demand_oldest_compatible_match environment.env_params.max_substitution_cost=650
python run_simopt.py +experiment=eight_product_perishable/eighth_demand_priority_match environment.env_params.max_substitution_cost=650