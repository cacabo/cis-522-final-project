import fsutils
from models.DeepRLModel import DeepRLModel

# Ensure that we can create a net and save it to the directory
model = DeepRLModel()
print('DeepRLModel', model)
print('Net', model.model)
fsutils.save_net_to_disk(model.model, 'test-fsutils')

