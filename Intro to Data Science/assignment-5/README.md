# How to Run
- First open or create a Python virtual environment. This can be done by following the commands below (in Bash):
```bash
python3 -m venv myvenv
source myvenv/bin/activate
```
If everything worked out correctly, you should see `(myvenv)` at the start of the prompt line.
- Then, install the required dependencies by running `pip install -r dependencies.txt`[^1]
- Make sure the `config.py` file lives on the same folder as `car.py`.
- Simply run the scripts by `python3 <script>.py`
# Reinforcement Learning Mini-Suite
This submission contains two "standalone" modules, namely `car.py` and `lake.py`. In the first one, the hyperparameters of the learning algorithm were stored into `config.py` so make sure to manipulate them from there. In contrast, `lake.py` was implemented in a much more compact way, so all the global variables are listed in the top of the script. Respectively, the `MountainCar-v0` and `FrozenLake-v1` reinforcement learning applications were implemented.
# Documentation - `car.py`
Training script for the mountain car application.
- Builds a **uniform bin grid** for position and velocity, by taking into consideration the hint on the assignment sheet (default 19 and 15 bins respectively). Then maps the continuous observation `[pos, vel]` to a **state index** using `s = i * VEL_BINS + j` where `np.digitize` takes place on the interior of the bin edges.
- Then it initializes a **Q-table** with shape `(POS_BINS * VEL_BINS, 3)` since the mountain car environment has 3 actions (according to its [documentation](https://gymnasium.farama.org/environments/classic_control/mountain_car/)). Prints its shape for sanity check verification.
- Runs the **Q-learning scheme** with update rule:
```python
Q[s,a] + alpha * (r + gamma * max_a' Q[s', a'] - Q[s, a]
```
see the `q_update` function for how this was implemented.
- Uses an epsilon greedy strategy, where with probability `EPSILON` we take a random action. Otherwise, we select one of the exploitation choices.
- This implementation overrides `gym`'s 200-step limit and sets a new (user-defined) limit as `MAX_STEPS`, when creating the environment.
- Logs per-episode **returns** and amount of **steps** in a `.csv` file, and also saves a **learning curve** `.png` in `output/`. A moving average curve is also showcased to help us humans understand what is going on.
# Documentation - `config.py`
The hyperparameters of `car.py`, modify them as you like, but take care when changing the `EPISODES` and `MAX_STEPS` variables. The reason is these live in a nested loop, so treat them with care.
# Documentation - `lake.py`
This script implements the Frozen Lake reinforcement learning scheme on a `4x4` game map. It uses the same **Q-learning** approach as `car.py`, since it was already coded once, I didn't find any value in doing something from scratch again. 

It prints a simple log of episodes and steps in the terminal, as well as a **policy map**, where the directions have been encoded by North, South, East, West. At the end, few episodes are rendered using the `pygame` package, so we can see what is actually going on. 

Hyperparameters for this reinforcement learning scheme are listed at the top of the file, and you can also change them to your heart's content. Obviously, not all combinations work.

[^1]: As always, this command will update the modules listed inside the dependency file to their latest versions. If you don't want this, make sure to switch from a global virtual environment to a local one.

