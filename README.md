# EE239AS

## Recreating Results

Ensure that package requirements are satisfied. See conda_requirements.txt for full list of packages installed for running. The given package list is for a linux environment (Ubuntu 18.04.4). Your mileage may vary. 

```bash
conda install --file conda_requirements.txt -c conda-forge
```
Some packages like OpenAI Gym may need to be installed individually using pip and/or conda.

To reproduce the results used in our paper, the included bash scripts will run training sessions with the desired hyper parameters.
Fine tuning of hyper parameters was done starting from `baseline_online_dqn.sh` and `baseline_online_ddqn.sh`. The best performing dqn we found used a learning rate of 0.0001 instead of the 0.001 in the baseline script. This can be run using `online_dqn_learning_rate_0001.sh`. For fair comparison, our ddqn training was based on the same settings but with varying number of episodes between saving the running network to the target network. These hyper parameters are defined in `online_ddqn_learning_rate_0001_copy_params_50.sh`, `online_ddqn_learning_rate_0001_copy_params_20.sh`, `online_ddqn_learning_rate_0001_copy_params_10.sh`, `online_ddqn_learning_rate_0001_copy_params_5.sh`, and `online_ddqn_learning_rate_0001_copy_params_2.sh`. These scripts prescribe that the running network be copied to the target network every 50, 20, 10, 5, and 2 episodes respectively. Where each episode includes on the order of 100 transitions. The rest of the hyper parameters are as follows:

* learning_rate=0.0001
* discount_factor=0.99
* use_ddqn=True
* batch_size=128
* max_replay_history=500000
* epsilon=0.995
* eval_episodes=16
* gd_optimizer=Adam
* decay=0.995

Where decay is used to decay the value of epsilon according to epsilon * (decay ^ i) where i is the elapsed number of iterations.

During training, statistics are saved every 5 iterations (variable parameter, see `python main.py --help`) to the `./runs/` directory. To see progress of training:
```
tensorboard --logdir=runs
```
Then visit `localhost:6006` on your desired web browser. Each run is named based on the start time of the model.


To see a trained model in action, run:
```
python visualize.py <MODEL_PATH>
```
Where MODEL_PATH is the path to the desired model. Models are automatically saved to `./models/<run_name>/<algorithm>_<iteration_number>.pt` where run_name is the start time of the run. The above bash files save models every 15 iterations. algorithm is either dqn or ddqn and iteration_number is the iteration the model was saved on.
