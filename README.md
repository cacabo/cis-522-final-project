# agar-py

Implementing agar.io to run locally as a Python process. This will optimize for our model being able to hook into the game state and learn in rapid iterations. We are developing a set of models to learn to play this game via Reinforcement Learning.

---

# TODO

### Done

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

### Up next

- [ ] Try dx and dy?
- [ ] Try stacking
- [ ] Try disallowing certain actions (it's not getting stuck on walls really anymore)
- [ ] Plot max reward over episodes
- [ ] Plot survival time over episodes
- [ ] Have encode agent state return a tensor, look where it is used (Cam)
- [ ] Remove time from state encoding, try radically simplifying things (Cam)
- [ ] Test state encoding, make sure it is seemingly correct (Cam)
- [ ] CNN hyperparams (Sam + Salib)

### Lower priority

- [ ] Smarter datastructure for checking collisions
- [ ] Move with mouse? Could move in directions other than the 8 we currently support
- [ ] Shooting viruses
