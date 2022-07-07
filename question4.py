import joblib
import pandas as pd
import numpy as np
import time
import warnings
import pdb
warnings.filterwarnings("ignore", category=UserWarning)

class GeneticGroup:
    def __init__(self,vector_reg,vector_clf,vector_v_index,dna_size,pd_variables_range,\
                    n_pop=500,crossover_rate=0.5,mutation_rate=0.1,\
                    n_generations=50,model_r=np.max,model_c1=np.min,\
                    model_c2=np.min,model_c3=np.min,model_c4=np.min,model_c5=np.min):
        self.v = vector_reg
        self.v_c = vector_clf
        self.vvi = vector_v_index
        self.m_reg = model_r
        self.m_c1 = model_c1
        self.m_c2 = model_c2
        self.m_c3 = model_c3
        self.m_c4 = model_c4
        self.m_c5 = model_c5
        self.dna_size = dna_size
        self.pop_num = n_pop
        self.cr = crossover_rate
        self.mr = mutation_rate
        self.ng = n_generations
        self.pvr = pd_variables_range
        self.feat_num = len(self.pvr)
        self.population = np.random.randint(2, size=(self.pop_num, self.dna_size * self.feat_num))
        self.x = np.tile(vector_reg,self.pop_num).reshape(self.pop_num,-1)
        self.x_clf = np.tile(vector_clf,self.pop_num).reshape(self.pop_num,-1)

    #进行解码操作
    def translateDNA(self):
        population = self.population
        #pdb.set_trace()
        data_x = np.zeros((self.pop_num, self.feat_num))
        for i in range(self.feat_num):
            pop_x = population[:,i::self.feat_num]
            #print('pop_x.shape:',pop_x.shape)
            #print('featurn counts:', self.feat_num)
            data_x[:,i] = pop_x.dot(2 ** np.arange(self.dna_size)[::-1]) / float(2 ** self.dna_size - 1)\
            *float(self.pvr['max'][i]-self.pvr['min'][i])+float(self.pvr['min'][i])
        self.x = data_x
        self.x_clf[:,self.vvi] = self.x
        return self.x,self.x_clf ##将二进制编码的值转换为实际的数值，这里要返回两个值，一个是回归用的数据，一个是分类用数据

    def get_fitness(self):
        data_x,data_xclf = self.translateDNA()
        # pdb.set_trace()
        pred_reg = self.m_reg.predict(data_x)
        total_ADMET = self.m_c1.predict(data_xclf) + self.m_c2.predict(data_xclf) +\
            (self.m_c3.predict(data_xclf)^1) + self.m_c4.predict(data_xclf) +\
            (self.m_c5.predict(data_xclf)^1)
        # pdb.set_trace()
        fitness_Reg = pred_reg - np.min(pred_reg) + 1e-4
        fitness_ADMET = (total_ADMET >= 4) * 1 + 1e-4
        return fitness_Reg * fitness_ADMET

    def select(self):
        idx = np.random.choice(np.arange(self.pop_num),size=self.pop_num,\
            replace=True,p=self.get_fitness()/np.sum(self.get_fitness()))
        self.population = self.population[idx]

    def mutation(self, child):
        if np.random.rand() < self.mr:
            mutate_point = np.random.randint(0,self.dna_size * self.feat_num)
            child[mutate_point] = child[mutate_point] ^ 1
    
    def crossover_and_mutation(self):
        population = self.population
        for i in range(self.pop_num):
            child = self.population[i,:]
            if np.random.rand() < self.cr:
                mother = population[np.random.randint(self.pop_num)]
                cross_points = np.random.randint(0, self.dna_size * self.feat_num)
                child[cross_points:] = mother[cross_points:]
            self.mutation(child)
            population[i,:] = child
        self.population = population
    
    def optimization(self):
        zy = np.zeros((self.ng,self.feat_num))
        for i in range(self.ng):
            self.select()
            self.crossover_and_mutation()
            fitness = self.get_fitness()
            if np.std(fitness) <= 1e-5:
                break
            zy[i,:] = self.x[int(np.argmax(fitness)),:]
        return self.x[int(np.argmax(fitness)),:],self.x_clf[int(np.argmax(fitness)),:],zy

print('---------------loading data----------------')
#加载基于随机森林的分类模型
RF_clf_Caco_2 = joblib.load('./model/forest_clf_Caco-2.pkl')
RF_clf_CYP3A4 = joblib.load('./model/forest_clf_CYP3A4.pkl')
RF_clf_hERG = joblib.load('./model/forest_clf_hERG.pkl')
RF_clf_HOB = joblib.load('./model/forest_clf_HOB.pkl')
RF_clf_MN = joblib.load('./model/forest_clf_MN.pkl')
#加载基于随机森林的回归预测模型
RF_reg = joblib.load('./model/forest_reg_y.pkl')
#加载要优化的回归数据
data_reg=pd.read_csv('./data/MD_train_20.csv',header=0).iloc[:,1:] #(1974,20)
label_reg = pd.read_excel('./data/ERα_activity.xlsx',header = 0).iloc[:,2]
#加载要优化的分类数据
data_clf=pd.read_csv('./data/MD_train_dropzero.csv',header=0).iloc[:,1:] #(1974,359)
label_clf= pd.read_excel('./data/ADMET.xlsx',header = 0)
Caco_2,CYP3A4,hERG,HOB,MN=label_clf.iloc[:,1],label_clf.iloc[:,2],label_clf.iloc[:,3],label_clf.iloc[:,4],label_clf.iloc[:,5]
print('shape of datareg:{}\tshape of data_clf:{}'.format(data_reg.shape,data_clf.shape))
data_s_v = pd.read_csv('./data/var_range.csv')
#编码长度
dna_size = int(np.ceil(np.log2(np.max((data_s_v['max']-data_s_v['min'])/data_s_v['delta']))))
print('dna_size:',dna_size)
print('length of data_reg:',len(data_reg))
## 回归用的20 个变量数据在分类用数据中的索引
vector_v_index = [32, 244, 314, 141, 16, 215, 242, 352, 199, 33, 15, 236, 294, 357, 18, 19, 17, 168, 346, 35]

#ADMET原本>=3的样本号
select_list=[8, 9, 11, 12, 15, 17, 18, 33, 50, 51, 59, 72, 73, 74, 77, 95, 98, 116, 120, 144, 148, 152, 178,\
            179, 180, 181, 184, 186, 188, 189, 190, 191, 234, 236, 237, 246, 248, 250, 260, 261, 296, 297, 327, 328,\
            335, 350, 406, 410, 412, 413, 415, 416, 427, 436, 443, 444, 445, 446, 449, 456, 459, 460, 461, 462, 463,\
            464, 465, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 484, 485, 488, 489, 490,\
            491, 498, 500, 501, 502, 503, 504, 506, 507, 508, 509, 510, 511, 512]
# 利用遗传算法获得优化的分子描述符的取值范围
#创建用来存储已优化的的特征数据,一个是ADMET，另一个是pIC50，用来反应算法的优化效果
res = np.zeros((len(select_list),len(data_reg.columns)+2)) 
res_clf = np.zeros((len(select_list),len(data_clf.columns)))

print('---------------training----------------')
for i in range(len(select_list)):
#for i in range(len(data_reg)):
    print('训练进度:',i)
    t1 = time.time()
    GG =GeneticGroup(np.array(data_reg.iloc[select_list[i],:]),np.array(data_clf.iloc[select_list[i],:]),\
        vector_v_index,dna_size,data_s_v,n_generations=500,model_r=RF_reg,\
        model_c1=RF_clf_Caco_2,model_c2=RF_clf_CYP3A4,model_c3=RF_clf_hERG,\
        model_c4=RF_clf_HOB,model_c5=RF_clf_MN)##创建一个实例化的对象
    res[i,:-2],res_clf[i,:],zy=GG.optimization()
    res[i,-2]=RF_reg.predict(res[i,:-2].reshape(1,-1)) ##保存模型预测的回归的具体值，方便与原始数据进行对比
    res[i,-1]=RF_clf_Caco_2.predict(res_clf[i,:].reshape(1,-1)) +\
        RF_clf_CYP3A4.predict(res_clf[i,:].reshape(1,-1)) +\
        (RF_clf_hERG.predict(res_clf[i,:].reshape(1,-1))^1)+\
        RF_clf_HOB.predict(res_clf[i,:].reshape(1,-1))+\
        (RF_clf_MN.predict(res_clf[i,:].reshape(1,-1))^1)
    t2 = time.time()
    print('第{}({})个样本优化过程消耗的时间：'.format(i,select_list[i]),t2-t1)
    print('第{}({})个样本优化后值：'.format(i,select_list[i]),res[i,-2],res[i,-1])
pd_res = pd.DataFrame(res)
pd_res.columns = list(data_reg.columns)+['pIC50','ADMET']
pd_res.to_csv('./data/ques4_sel0.csv',index=False)
###保存第一个样本的迭代过程
# pd_zy = pd.DataFrame(zy)
# pd_zy.columns = list(data_reg.columns)
# pd_zy.to_excel('./data/' + str(i) + '号样本的优化过程.xlsx',index=False)