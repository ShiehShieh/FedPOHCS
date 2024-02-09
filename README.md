# FedKL
===================================

Dependencies
------------

- Python

Please refer to requirements.txt. Dependencies can be installed by using the following command:

    pip install requirements.txt

- Bazel

Please install [Bazel](https://bazel.build/install). After installing it, you can build the program with

    bazel build -c opt //...

and run it with

    ./main ...



File Structure
------------

- Algorithm and Model Implementations: model/
    - RL related: model/fl
    - FL related: model/rl
        - Agent wrapper: model/rl/agent.py
        - The core of local actor: model/rl/trpo.py
        - Critic: model/rl/critic.py
    - Optimizers: model/optimizer
- Customized RL Environments: environment/
- Implementation of federated client/device: client/


Example Usage
-----

To reproduce the Hoppers experiment in our [FedPOHCS](https://arxiv.org/abs/2305.10978) paper:

    ./main --pg=TRPO --fed=FedAvg --lr=3e-2 --kl_targ=3e-3 --ent_coef=1e-1 --ent_decay=5 --lr_decay=0.9 --distance_metric=sqrt_kl --retry_min=-500 --n_local_iter=20 --parallel=10 --clients_per_round=6 --num_cands=18 --heterogeneity_type=dynamics --expose_critic --env=hopper --init_seed=1 --cs=random --svf_n_timestep=2e5 --disable_retry --b_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-random-b.csv --da_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-random-da.csv --avg_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-random-avg.csv --reward_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-random-evalh.csv --parti_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-random-parti.csv --cands_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-random-cands.csv
    ./main --pg=TRPO --fed=FedAvg --lr=3e-2 --kl_targ=3e-3 --ent_coef=1e-1 --ent_decay=5 --lr_decay=0.9 --distance_metric=sqrt_kl --retry_min=-500 --n_local_iter=20 --parallel=10 --clients_per_round=6 --num_cands=18 --heterogeneity_type=dynamics --expose_critic --env=hopper --init_seed=2 --cs=random --svf_n_timestep=2e5 --disable_retry --b_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-random-b.csv --da_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-random-da.csv --avg_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-random-avg.csv --reward_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-random-evalh.csv --parti_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-random-parti.csv --cands_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-random-cands.csv

    ./main --pg=TRPO --fed=FedAvg --lr=3e-2 --kl_targ=3e-3 --ent_coef=1e-1 --ent_decay=5 --lr_decay=0.9 --distance_metric=sqrt_kl --retry_min=-500 --n_local_iter=20 --parallel=10 --clients_per_round=6 --num_cands=18 --heterogeneity_type=dynamics --expose_critic --env=hopper --init_seed=0 --cs=csh-max --eval_heterogeneity --svf_n_timestep=2e5 --disable_retry --b_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-b.csv --da_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-da.csv --avg_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-avg.csv --reward_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-evalh.csv --parti_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-parti.csv --cands_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-cands.csv
    ./main --pg=TRPO --fed=FedAvg --lr=3e-2 --kl_targ=3e-3 --ent_coef=1e-1 --ent_decay=5 --lr_decay=0.9 --distance_metric=sqrt_kl --retry_min=-500 --n_local_iter=20 --parallel=10 --clients_per_round=6 --num_cands=18 --heterogeneity_type=dynamics --expose_critic --env=hopper --init_seed=1 --cs=csh-max --eval_heterogeneity --svf_n_timestep=2e5 --disable_retry --b_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-b.csv --da_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-da.csv --avg_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-avg.csv --reward_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-evalh.csv --parti_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-parti.csv --cands_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-cands.csv
    ./main --pg=TRPO --fed=FedAvg --lr=3e-2 --kl_targ=3e-3 --ent_coef=1e-1 --ent_decay=5 --lr_decay=0.9 --distance_metric=sqrt_kl --retry_min=-500 --n_local_iter=20 --parallel=10 --clients_per_round=6 --num_cands=18 --heterogeneity_type=dynamics --expose_critic --env=hopper --init_seed=2 --cs=csh-max --eval_heterogeneity --svf_n_timestep=2e5 --disable_retry --b_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-b.csv --da_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-da.csv --avg_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-avg.csv --reward_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-evalh.csv --parti_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-parti.csv --cands_history_fn=fedavg-nonlinear-hopper-6c-18d-20iter-sqrt_kl-3e-2-3e-3-1e-1-5-0.9-dynamics-norm-mu0-std0.2-ec-seed0-cshmax-cands.csv

Type `./main --help` for a list of all key flags.

## References
Please refer to our [FedPOHCS](https://arxiv.org/abs/2305.10978) paper for more details as well as all references.
