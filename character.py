import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
import matplotlib.image as mpimg
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import seaborn as sns
import pdb


Molecular_Descriptor=pd.read_csv('./data/MD_train_dropzero.csv',header=0)
Molecular_Descriptor=Molecular_Descriptor.iloc[:,1:]
print(Molecular_Descriptor.shape)

ERa_activity= pd.read_excel('./data/ERα_activity.xlsx',header = 0)
ERa_activity=ERa_activity.iloc[:,2]
print(ERa_activity.shape)

#随机森林降维
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rnd_clf.fit(Molecular_Descriptor, ERa_activity.astype('int'))
rnd_clf.feature_importances_.shape

#排序前贡献率图
x_data = rnd_clf.feature_importances_
plt.figure(figsize=(10,5.5))
plt.bar(x = range(0,len(x_data)),height = x_data,align='center',color='b')
plt.xlabel('feature',labelpad = 19) # 控制标签和坐标轴的距离
plt.ylabel('contribution',labelpad =10)
plt.savefig('./pic/contribution.png', dpi=500, bbox_inches='tight') # 解决图片不清晰，不完整的问题
plt.show()

#排序后贡献率图
sorted_id = sorted(range(len(x_data)), key=lambda k: x_data[k], reverse=True)
feature_importances_descend=sorted(rnd_clf.feature_importances_,reverse=True)
x = np.arange(0,len(feature_importances_descend))
plt.figure(figsize=(12,9), dpi= 80)
plt.xlabel('feature',labelpad = 19) # 控制标签和坐标轴的距离
plt.ylabel('contricbution',labelpad =10)
plt.plot(x,feature_importances_descend)
plt.savefig('./pic/contribution_rank.png', dpi=500, bbox_inches='tight') # 解决图片不清晰，不完整的问题
plt.show()

#30个特征相关性图
index_id_30=sorted_id[0:30]
# header=list(Molecular_Descriptor.columns)
# header_30=[]
# for i in range(30):
#     header_30.append(header[index_id_30[i]])
# print(index_id_30)
# print(header_30)
Molecular_Descriptor_30=Molecular_Descriptor.iloc[:,index_id_30]
print(Molecular_Descriptor_30.shape)

# Plot
plt.figure(figsize=(17,14), dpi= 80)
sns.heatmap(Molecular_Descriptor_30.corr(),\
xticklabels=Molecular_Descriptor_30.corr().columns,\
yticklabels=Molecular_Descriptor_30.corr().columns, cmap='YlGnBu', center=0, annot=True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('./pic/correlation_30.png', dpi=500, bbox_inches='tight')
plt.show()

#20 个特征相关性图
index_id_20=[32, 244, 314, 141, 16, 215, 242, 352, 199, 33,\
15, 236, 294, 357, 18, 19, 17, 168, 346, 35]
Molecular_Descriptor_20=Molecular_Descriptor.iloc[:,index_id_20]
# for i in range(20):
#     print(header[index_id_20[i]])
# for i in range(20):
#     print(x_data[index_id_20[i]])
print(Molecular_Descriptor_20.shape)

# Plot
plt.figure(figsize=(17,14), dpi= 500)
sns.heatmap(Molecular_Descriptor_20.corr(),
xticklabels=Molecular_Descriptor_20.corr().columns,
yticklabels=Molecular_Descriptor_20.corr().columns, cmap='YlGnBu', center=0, annot=True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('./pic/correlation_20.png', dpi=500, bbox_inches='tight') # 解决图片不清晰，不完整的问题
plt.show()

# save
MD20=pd.read_csv('./data/MD_train_dropzero.csv',header=0)
MD20=MD20.iloc[:,[0, 33, 245, 315, 142, 17, 216, 243, 353, 200, 34,\
16, 237, 295, 358, 19, 20, 18, 169, 347, 36]]
MD20.to_csv('./data/MD_train_20.csv',header=True,index=False)