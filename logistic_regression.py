import numpy as np

#生成数据209x64x64
train_x=np.random.randint(1,255,size=(209,64,64)).reshape(-1,209)/255.0
#生成标签
train_y=np.random.randint(0,2,size=(1,209)).reshape(1,209)

#划分训练集测试集
train_x_set=train_x[:,:150]
train_y_set=train_y[:,:150]
test_x_set=train_x[:,150:]
test_y_set=train_y[:,150:]

#初始化
def initial(dim):
    w=np.zeros((dim,1))
    b=np.zeros(1)
    return w,b

#定义逻辑函数
def sigmoid(x):
    X=1/(1+np.exp(-x))
    return X

#后向传播,计算 dw,db,cost
def propogate(x,y,w,b):
    Z=np.dot(w.T,x)+b
    a=sigmoid(Z)
    assert(a.shape==y.shape)
    #计算成本函数
    m=x.shape[1]
    
    cost=-1/m*np.sum(y*np.log(a)+(1-y)*np.log(1-a))
    cost=cost.reshape(-1)
    dw=1/m*np.dot(x,(a-y).T)
    assert(dw.shape==w.shape)
    db=1/m*np.sum(a-y)
    assert(db.dtype==float)
    return dw,db,cost

#定义优化器
def optimize(train_x_set,train_y_set,w,b,num_iteration,learning_rete):
    costs=[]
    for i in range(num_iteration):
        dw,db,cost=propogate(train_x_set,train_y_set,w,b)
        
        w=w-learning_rete*dw
        b=b-learning_rete*db

        if i%100==0:
            costs.append(cost)
            print('iteration'+str(i)+':'+str(cost))
    return w,b,costs

#进行预测
def prediction(test_x_set,w,b):
    dim=test_x_set.shape[1]
    y_prediction=np.zeros((1,dim))
    w.reshape(test_x_set.shape[0],1)
    A=sigmoid(np.dot(w.T,test_x_set)+b)
    
    for i in range(dim):
        if A[0,i]>0.5:
            y_prediction[0,i]=1
        else:
            y_prediction[0,i]=0
    
    return y_prediction

#logistic regression model
def model(train_x_set,train_y_set,test_x_set,test_y_set,num_iteration,learning_rate):
    test_y_set.reshape(1,-1)
    dim=train_x_set.shape[0]
    w,b=initial(dim)
    w,b,costs=optimize(train_x_set,train_y_set,w,b,num_iteration,learning_rate)
    
    y_prediction_train=prediction(train_x_set,w,b)
    y_prediction=prediction(test_x_set,w,b)
    
    acc_train=np.mean(np.abs(y_prediction_train-train_y_set))
    acc_test = np.mean(np.abs(y_prediction-test_y_set))
    print('accuracy of train:'+str(acc_train))
    print('accuracy of test:'+str(acc_test))
    d={
        'w':w,
        'b':b,
        'num_iteration':num_iteration,
        'learning_rate':learning_rate,
        'y_predictin_test':y_prediction,
        'y_prediction_train':y_prediction_train
    }
    return d
    
    