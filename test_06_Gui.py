from functions import *
import pandas as pd

## Import the data
data = np.loadtxt('Xtr.csv', delimiter = ',')
y = pd.read_csv('Ytr.csv')
y = list(y['Prediction'])

###########
## Xtr
##########

# Make 2 new train sets by rotating Xtr
# Rotations: +20° and -20°

new_data = rotate_dataset(data, 20)
new_data2 = rotate_dataset(data, -20)

# Solve problem with size
new_data = new_data[:,:,0] 
new_data2 = new_data2[:,:,0]

# Concatenate the 3 data sets
big_data = np.concatenate([data, new_data, new_data2], axis = 0)
big_data.shape # (15 000L, 784 L)

# Store the new Xtr into "Xtr_bigdata.csv"
np.savetxt('Xtr_bigdata.csv', big_data)

############
### Ytr
############

# Extend the Ytr
y_new = y[:]
y_new.extend(y)
y_new.extend(y)

len(y) # 15 000L

# Store the new Ytr into "Ytr_bigdata.csv"
out = pd.DataFrame(data={"Id":[i for i in range(1,len(y_new)+1)], "Prediction": y_new})
out["Prediction"] = out["Prediction"].values.astype('int')
out.to_csv('Ytr_bigdata.csv', index = False, quoting = 3)