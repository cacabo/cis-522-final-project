# agar-py

Implementing agar.io to run locally as a Python process. This will optimize for our model being able to hook into the game state and learn in rapid iterations. We are developing a set of models to learn to play this game via Reinforcement Learning.

TODO

---

### Repo structure

```
.
├── .vscode/                     Editor config
├── __pycache__/                 Cached build files
|
├── model_utils/                 Code shared by models
|   └── ReplayBuffer.py          Replay buffer datastructure
|
├── models/                      Agar.io models
|   ├── DeepCNNModel.py          CNN-based RL agent on screenshot state
|   ├── DeepRLModel.py           FC-based RL agent on encoded state
|   ├── HeuristicModel.py        Greedy baseline model
|   ├── ModelInterface.py        Shared interface for all models
|   └── RandomModel.py           Random action baseline model
|
├── notebooks/                   Set of notebooks used for training on Sagemaker
|   ├── cam-notebook.ipynb       There is one for each team member
|   ├── daniel-notebook.ipynb
|   ├── mak-notebook.ipynb
|   └── sam-notebook.ipynb
|
├── plots/
|   ├── important/               Best performing plots
|   └── *                        Plots generated over the course of training and test
|
├── store/
|   └── nets/                    Nets generated over the course of training
|       ├── important/           Best performing nets
|       └── *                    Nets generated over the course of training
|
├── __test-encoded-state__.py    Tests that RL agenet state encoding works as expected
├── __test-fsutils__.py          Tests that saving and loading net params works as expected
|
├── actions.py                   Set of allowed actions for an agent to take
├── agario.py                    Script for running game manually in GUI
├── agent.py                     Agent and AgentCell classes
├── camera.py                    Handles moving viewport in GUI to follow agent
├── config.py                    Set of constants used throughout the repo
├── evaluate.py                  Plots performance of set of models
├── food.py                      Food object
├── fsutils.py                   Helper functions for saving
├── gamestate.py                 Implementation of headless game state
├── mass.py                      Mass pellet object
├── test.py                      Run trained model on GUI
├── train_cnn.py                 Train CNN model
├── train.py                     Train RL model
├── trainutil.py                 Shared functions by the two training scripts
├── utils.py                     Helper functions for dealing with object interactions
├── virus.py                     Virus object
|
├── requirements.txt             Python dependencies
├── .gitignore                   Files to not include in git
└── README.md                    Documentation
```

---

### Running the code

First, clone this repo and change directories into the root of the repo.

#### Installing dependencies

You need python3 (and pip3) installed to run our project.

To install dependencies, run:

```bash
pip3 install -r requirements.txt
```

TODO

TODO pygame especially

#### Running locally

TODO

#### Running on AWS Sagemaker

TODO

---

### Tasks

#### Done

- [x] Split via space bar
- [x] Shoot via `q` as we are currently using `w` for motion
- [x] Viruses splitting agents which are signficantly larger than the virus
- [x] Recombining agent cell parts
- [x] Eventually merge split cells back together
- [x] Have some intertia to bring cells together
- [x] Finish acceleration (Cam)
- [x] Make it such that an agent cannot split after a certain number of clock ticks not ms
- [x] Checks to see if AgentCells are overlapping
- [x] Eating mass
- [x] Split into many pieces when eats virus
- [x] Case where cells totally overlap (pick random angle)
- [x] Bug with eating parts of agents
- [x] Fix bug with movement of heuristic agent at edge of map (Sam)
- [x] Get bigger at edge of map -> should get scooched back into map boundary (Sam)
- [x] When you die you keep getting negative reward -> this should only happen once (Mak)
- [x] Create IAM users for each group member on ccabo@seas AWS account
- [x] Add GUI functionality to the training loop (Sam)
- [x] Mass decay (cam)
- [x] Infrastructure for saving model net params (maybe to a folder in github?) (cam)
- [x] Update rewards function to penalize being eaten (Mak)
- [x] Privacy.com on AWS account & billing alerts (Cam)
- [x] Setup infra on AWS EC2 (Salib)
- [x] Try running it locally and then via sagemaker in a notebook (Salib)
- [x] Refactor train file to not run on import (Mak)
- [x] Optimize state encoding runtime (Cam)
  - [x] Look into encoding state via pytorch -> distributed
- [x] Store encoded state in buffer; make sure this is not by reference (mak)
- [x] Random actions until replay buffer is full (Mak)
  - [x] Make sure to print at useful points (like when buffer is full and we start training)
  - [x] Don't learn until the buffer is full
- [x] Convolutional jawn (Sam + Salib)
  - [x] Rip the CNN
  - [x] Model
  - [x] Getting screenshots in the train loop (not just GUI)
- [x] Analyze RL runtime bottleneck (local v. AWS v. GPU) (Mak)
- [x] target net policy net (mak)
- [x] Add one more linear layer and make it normal sizes (Mak)
- [x] epsilon decay over episodes (mak)
- [x] Better parameterization for epsilon decay (and other hyperparams, ex. decay over) (Cam)
- [x] Test different state encodings (Cam & Mak)
- [x] Remove time from state encoding, try radically simplifying things (Cam)
- [x] Try disallowing certain actions (it's not getting stuck on walls really anymore)
- [x] CNN hyperparams (Sam + Salib)

#### Up next

- [ ] Try dx and dy
- [ ] Try stacking
- [ ] Plot max reward over episodes
- [ ] Plot survival time over episodes

#### Lower priority

- [ ] Smarter datastructure for checking collisions
- [ ] Move with mouse? Could move in directions other than the 8 we currently support
- [ ] Shooting viruses
