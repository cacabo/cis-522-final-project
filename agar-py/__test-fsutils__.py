import fsutils
import pandas as pd
from models.DeepRLModel import DeepRLModel

# Ensure that we can create a net and save it to the directory
model = DeepRLModel()
print('DeepRLModel', model)
print('Net', model.model)
fsutils.save_net_to_disk(model.model, 'test-fsutils')

# Ensure we can crate a datafram and save it to the directory
df = pd.DataFrame([[1, 2, 3]])
print(df)
fsutils.save_df_to_disk(df, 'test-fsutils')
