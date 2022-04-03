import numpy as np # Linear Algebra
import pandas as pd # Data Processing
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
# import model yang ingin difit
from sklearn.linear_model import LogisticRegression
# split data
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/user/Downloads/archive/SaYoPillow.csv')
df.columns = ['laju dengkuran', 'laju pernapasan',
           'suhu tubuh', 'gerakan anggota badan',
           'oksigen darah', 'gerakan mata',
           'jam tidur', 'detak jantung','stress level']

class Model:
    def __init__(self, data):
        self.data = data
        self.intro()
    def relation(self):
        self.corr = self.data.corr()
        self.corr.style.background_gradient(cmap='coolwarm')
        print("Hubungan : \n{}".format(self.corr))
    def info(self):
        print(self.data.info())
    def comparisonplot(self, col1, col2, target):
        sns.scatterplot(data=self.data, x=col1, y=col2, hue=target)
        plt.show()
    def relationplot(self):
        self.cols = []
        for i in self.data.columns:
            self.cols.append(i)
            for j in self.data.columns:
                if j not in self.cols:
                    self.comparisonplot(i, j, 'stress level')
    def intro(self):
        self.x = self.data.drop('stress level', axis=1)
        self.y = self.data['stress level']
        self.scaler = MinMaxScaler()
        self.x_data_scaled = pd.DataFrame(self.scaler.fit_transform(self.x), columns=self.x.columns)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, random_state=42)
        self.scaler.fit(self.x_train)

        self.x_train_scaled = self.scaler.transform(self.x_train)
        self.x_test_scaled = self.scaler.transform(self.x_test)

        self.log = LogisticRegression().fit(self.x_train_scaled, self.y_train)
        
    def info_scale(self):
        print(self.x_data_scaled)
    def shape(self):
        print(self.x_train.shape, self.x_test.shape, self.y_train.shape, self.y_test.shape)
        
