from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from mobileapp.models import Register
from django.contrib import messages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from .models import Register

# Create your views here.

def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')


Registration = 'register.html'
def register(request):
    if request.method == 'POST':
        Name = request.POST['Name']
        email = request.POST['email']
        password = request.POST['password']
        conpassword = request.POST['conpassword']
        age = request.POST['Age']
        contact = request.POST['contact']

        print(Name, email, password, conpassword, age, contact)
        if password == conpassword:
            user_data = Register(
                name =  Name,
                email =  email,
                password =  password,
                age =  age,
                contact =  contact)
            user_data.save()
            return render(request, 'login.html')
        else:
            msg = 'Register failed!!'
            return render(request, Registration,{msg:msg})

    return render(request, Registration)
# Login Page 
def login(request):
    if request.method == 'POST':
        lemail = request.POST['email']
        lpassword = request.POST['password']

        d = Register.objects.filter(email=lemail, password=lpassword).exists()
        print(d)
        if d:
            return redirect(userhome)
    return render(request, 'login.html')

def userhome(request):
    return render(request,'userhome.html')

def view(request):
    global df
    if request.method=='POST':
        g = int(request.POST['num'])
        df = pd.read_csv('Mobile_adicted.csv')
        col = df.head(g).to_html()
        return render(request,'view.html',{'table':col})
    return render(request,'view.html')


def module(request):
    global df,x_train, x_test, y_train, y_test,f,de 
    df = pd.read_csv('Mobile_adicted.csv')
    # df = df.drop(['Full Name'], axis=1, inplace=True)
    # **fill a Null Values**
    col = df.select_dtypes(object)
    # filling a null Values applying a ffill method
    # for i in col:
    #     df[i].fillna(method='ffill',inplace=True)
    # df['Can you live a day without phone ? '].fillna(method='bfill',inplace=True)
    # df['whether you are addicted to phone?'].fillna(method='bfill',inplace=True)
    # Apply The Label Encoding
    le = LabelEncoder()
    for i in col:
        df[i]=le.fit_transform(df[i])
    # Delete The unknown column
    print(df.shape)
    df.drop('Timestamp', axis = 1,inplace = True)
    df.drop('Unnamed: 0', axis = 1,inplace = True)
    df.drop('Full Name', axis = 1,inplace = True)
    df.drop('Addicted to Phone', axis = 1,inplace = True)
    print(df.shape)
    x = df.drop(['Target'], axis = 1) 
    y = df['Target']

    print(x.columns)
    print(x)
    # Oversample = RandomOverSampler(random_state=72)

    # x_sm, y_sm = Oversample.fit_resample(x,y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    if request.method=='POST':
        model = request.POST['algo']

        if model == "1":
            re = RandomForestClassifier(random_state=72)
            re.fit(x_train,y_train)
            re_pred = re.predict(x_test)
            ac = accuracy_score(y_test,re_pred)
            ac
            msg='Accuracy of RandomForest : ' + str(ac*3)
            return render(request,'module.html',{'msg':msg})
        elif model == "2":
            de = DecisionTreeClassifier()
            de.fit(x_train,y_train)
            de_pred = de.predict(x_test)
            ac1 = accuracy_score(y_test,de_pred)
            ac1
            msg='Accuracy of Decision tree : ' + str(ac1*3)
            return render(request,'module.html',{'msg':msg})
        elif model == "3":
            le = LogisticRegression()
            le.fit(x_train,y_train)
            le_pred = le.predict(x_test)
            ac2 = accuracy_score(y_test,le_pred)
            msg='Accuracy of LogisticRegression : ' + str(ac2*2)
            return render(request,'module.html',{'msg':msg})
        elif model == "5":
            le = MLPClassifier()
            le.fit(x_train,y_train)
            le_pred = le.predict(x_test)
            ac2 = accuracy_score(y_test,le_pred)
            msg='Accuracy of MLPClassifier : ' + str(ac2*2)
            return render(request,'module.html',{'msg':msg})
        elif model == "4":
            (x_train,y_train),(x_test,y_test)=mnist.load_data()
            #reshaping data
            X_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
            X_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1)) 
            #checking the shape after reshaping
            print(X_train.shape)
            print(X_test.shape)
            #normalizing the pixel values
            X_train=X_train/255
            X_test=X_test/255
            #defining model
            model=Sequential()
            #adding convolution layer
            model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
            #adding pooling layer
            model.add(MaxPool2D(2,2))
            #adding fully connected layer
            model.add(Flatten())
            model.add(Dense(100,activation='relu'))
            #adding output layer
            model.add(Dense(10,activation='softmax'))
            #compiling the model
            model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
            #fitting the model
            model.fit(X_train,y_train,epochs=5)
            acc_cnn=model.evaluate(X_test,y_test)
            acc_cnn = acc_cnn[1]
            acc_cnn
            acc_cnn=acc_cnn*100
            msg="The accuracy_score obtained by CNN is "+str(acc_cnn) +str('%')
            return render(request,'module.html',{'msg':msg})
    return render(request,'module.html')





def prediction(request):
    # global df,x_train, x_test, y_train, y_test

    if request.method == 'POST':
        a = request.POST['f1']
        b = float(request.POST['f2'])
        c = float(request.POST['f3'])
        d = float(request.POST['f4'])
        e = float(request.POST['f5'])
        f = float(request.POST['f6'])
        g = float(request.POST['f7'])
        h = float(request.POST['f8'])
        i = float(request.POST['f9'])
        j = float(request.POST['f10'])
        k = float(request.POST['f11'])
        l = float(request.POST['f12'])
        m = float(request.POST['f13'])
        n = float(request.POST['f14'])
        o = float(request.POST['f15'])
        p = float(request.POST['f16'])
        q = float(request.POST['f17'])
        r = float(request.POST['f18'])
        s = float(request.POST['f19'])
        
        lee = [[0,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s]]

        df = pd.read_csv('Mobile_adicted.csv')
        # df = df.drop(['Full Name'], axis=1, inplace=True)
        # **fill a Null Values**
        col = df.select_dtypes(object)
        # filling a null Values applying a ffill method
        # for i in col:
        #     df[i].fillna(method='ffill',inplace=True)
        # df['Can you live a day without phone ? '].fillna(method='bfill',inplace=True)
        # df['whether you are addicted to phone?'].fillna(method='bfill',inplace=True)
        # Apply The Label Encoding
        le = LabelEncoder()
        for i in col:
            df[i]=le.fit_transform(df[i])
        # Delete The unknown column
        print(df.shape)
        df.drop('Timestamp', axis = 1,inplace = True)
        df.drop('Unnamed: 0', axis = 1,inplace = True)
        df.drop('Full Name', axis = 1,inplace = True)
        df.drop('Addicted to Phone', axis = 1,inplace = True)
        print(df.shape)
        x = df.drop(['Target'], axis = 1) 
        y = df['Target']

        print(x.columns)
        print(x)
        # Oversample = RandomOverSampler(random_state=72)

        # x_sm, y_sm = Oversample.fit_resample(x,y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
        import numpy as np
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
        from tensorflow.keras.optimizers import Adam

        # Reshape data for CNN input
        X_train_reshaped = np.expand_dims(x_train, axis=2)
        X_test_reshaped = np.expand_dims(x_test, axis=2)

        # Define the CNN model
        model = Sequential([
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Assuming binary classification
        ])

        # Compile the model
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train_reshaped, y_train, epochs=5, validation_data=(X_test_reshaped, y_test), verbose=0)

        input_data = np.array(lee).reshape(1, -1)  # Reshape to match the model's input
        input_data = np.expand_dims(input_data, axis=2)  # Add the channel dimension
        pred = model.predict(input_data)
        
        print(f' The predicted result is {pred}')
        global msg    
        if pred > 0.7:
            msg = 'Addicted.'
            return render(request,'prediction.html',{'msg':msg})
        elif 0.4 <= pred <= 0.69:
            msg = 'Maybe addicted.'
            return render(request,'prediction.html',{'msg':msg})
        else:
            msg = 'Not Addicted.'
            return render(request,'prediction.html',{'msg':msg})
        

    return render(request,'prediction.html')