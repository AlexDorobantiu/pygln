# %%
import numpy as np
from sklearn.metrics import accuracy_score
from datetime import datetime
from pygln.numpy import PaperLearningRate
from pygln.numpy import GLN
from pygln.utils import get_mnist

X_train, y_train, X_test, y_test = get_mnist()

model10 = GLN(layer_sizes=[4, 4, 1], context_map_size = [4, 2, 1],
 input_size=X_train.shape[1], num_classes=10, learning_rate=PaperLearningRate(), bias = False, context_bias = False)

print("Steps: ", X_train.shape[0])
numberOfCorrectTrainingPredictions = 0
for n in range(X_train.shape[0]):
    predictedLabel = model10.predict(input=X_train[n:n+1], target=y_train[n:n+1])
    if predictedLabel == y_train[n]:
        numberOfCorrectTrainingPredictions += 1
    if n % 1000 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")        
        print(current_time, "Step:", n, "Accuracy:", (numberOfCorrectTrainingPredictions / (n+1)))


# %%
preds = []
print("Steps: ", X_test.shape[0])
for n in range(X_test.shape[0]):
    if n % 1000 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")        
        print(current_time, "Step:", n)
    preds.append(model10.predict(X_test[n]))

accuracy_score(y_test, np.vstack(preds))

# %%
