# agar-py

Implementing agar.io to run locally as a Python process. This will optimize for our model being able to hook into the game state and learn in rapid iterations.

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
- [x] Fix food bug (all the boys 😤)

### Up next

- [x] Add GUI functionality to the training loop (Sam)
- [x] Mass decay (cam)
- [ ] Look into ngrok for logging from AWS (cam)
- [ ] Infrastructure for saving model net params (maybe to a folder in github?) (cam)
- [x] Update rewards function to penalize being eaten (Mak)
- [ ] Training loop for basic RL agent (Mak + Cam)
  - [ ] Plug in the state encoding
  - [ ] Analyze runtime bottleneck
  - [ ] Play around with different state encodings and compare performance
- [ ] Convolutional jawn (Sam + Salib)
  - [ ] Model
  - [ ] Getting screenshots in the train loop (not just GUI)
    - [ ] Figure out a good frequency for this
- [ ] [Setup infra on AWS EC2 (Salib)](https://piazza.com/class/k58sba3uizm5md?cid=600)
- [ ] Try running it locally and then via sagemaker in a notebook (Salib)

### Lower priority

- [ ] Smarter datastructure for checking collisions
- [ ] Move with mouse? Could move in directions other than the 8 we currently support
- [ ] Shooting viruses
- [ ] safe guards for when you're dead you don't have actions
  - currently assumes game state properly keeps track which is ok i guess
  - edge case is manualy start a game where agentmodel.done = True
- [ ] Figure out what logs / graphs we will want on the writeup
