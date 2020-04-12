# agar-py

Implementing agar.io to run locally as a Python process. This will optimize for our model being able to hook into the game state and learn in rapid iterations.

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

### Up next

- [ ] Fix bug with movement of heuristic agent at edge of map
- [ ] Get bigger at edge of map -> should get scooched back into map boundary
- [ ] When you die you keep getting negative reward -> this should only happen once
- [ ] Training loop for basic RL agent (Mak + Cam)
  - [ ] Plug in the state encoding
  - [ ] Play around with different state encodings and compare performance
- [ ] Convolutional jawn
- [ ] Training on AWS EC2

### Lower priority

- [ ] Smarter datastructure for checking collisions
- [ ] Move with mouse? Could move in directions other than the 8 we currently support
- [ ] Shooting viruses