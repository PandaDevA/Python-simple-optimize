#%%
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats

tp_data = pickle.load(open("tp_all_.pkl", "rb"))
fp_data1 = pickle.load(open("fp_all_.pkl", "rb"))
tp_data1 = pickle.load(open("tp_all_1.pkl", "rb"))
fp_data = pickle.load(open("fp_all_1.pkl", "rb"))

def precision_line(b, m, tp_all_, fp_all_):
  count_tps=tp_all_.shape[0]
#   count_fps=fp_all_.shape[0]
#   print(count_tps)
#   print(count_fps)
  tp_above_line = len(np.where(tp_all_[:,1] >= tp_all_[:,0] * m + b)[0])
  fp_above_line = len(np.where(fp_all_[:,1] >= fp_all_[:,0] * m + b)[0])
#   print(tp_above_line)
#   print(fp_above_line)
  tps = tp_above_line/count_tps
#   fps = fp_above_line/count_fps
  precision =tp_above_line/(tp_above_line+fp_above_line)
#   print("precision:", precision, tps)
  return precision, tps#, fps

max_intercept = max(min(tp_data[:, 0]) * -1, max(tp_data[:, 1]))
min_intercept = 0
# print(min(tp_data[:, 0]), max(tp_data[:, 1]))
print("b_max:", max_intercept)

density = 0.01


# extract optimize intercept by rating of precision & tps
while(max_intercept - min_intercept > density) :
  interval = (max_intercept - min_intercept) / 10
  for idx in np.arange(min_intercept, max_intercept + interval, interval) :
    # print("b_step:", idx)
    result = precision_line(idx, 1, tp_data, fp_data)

    # make decision by importance between precision & tps
    decision = result[1]  - result[0] * 0.4

    if (decision < 0) :
      max_intercept = idx
      min_intercept = idx - interval
      break

#   print(max_intercept, min_intercept)
#   print("========")
print("optimized intercept:", max_intercept)
print("optimized stat:", precision_line(max_intercept, 1, tp_data, fp_data))

# ===========    
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.scatter(tp_data[:, 0], tp_data[:, 1], s=1)
plt.scatter(fp_data[:, 0], fp_data[:, 1], s=1)

slope = 1
intercept = max_intercept

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, tp_data[:, 0]))
plt.plot(tp_data[:, 0], mymodel)
plt.show()


# ===========
# x = np.random.normal(170, 10, 250)

# plt.hist(x)
# plt.show()
