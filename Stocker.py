import pandas as pd
import time
import datetime
import numpy as np

#plot function
import matplotlib.pyplot as plt
import matplotlib as mpl

import copy
from sklearn.preprocessing import MinMaxScaler
import statsmodels.tsa.api as smt
#measurement function
from sklearn.metrics import mean_squared_error

class stocker:
    '''this class is designed for each stock, it can plot and compare results from prediction'''
    def __init__(self, seed=9001):
        '''the initialize takes argument of random split seed, which will be used when we do random split'''
        self.read = -1
        self.seed = seed
        self.data = None
        self.target = None #the stock we are checking
        self.test_idx = None #we need to store test idx once we run split in order to plot them in the same graph
        self.splited = -1 #whether we have finished split data
        self.random = None #this is to indicate which split rule we use because it should create different plot style
        self.X_scaler = None
        self.y_scale = None
        
        
    def edit_seed(self, num):
        self.seed=num
    
    def import_data(self,data):
        '''this method will inport the data downloaded from a downloader object'''
        if not isinstance(data,pd.core.frame.DataFrame):
            raise ValueError("You should input data from a downloader object and call the pull_stock_parllel function")
        self.data = data
        self.target = list(data.columns)[1] 
        #now import data is finished, and we assign read to 1
        print("import stock data from " + self.target)
        self.read = 1
    
    def plot_self(self):
        '''this function will plot the one target stock saved already'''
        if self.read == -1:
            raise ValueError("You didn't import_data")
        stock=self.target
        #now we generate the plot
        plt.figure(figsize=(14, 5), dpi=100)
        plt.plot(self.data['Date'], self.data[stock], label=stock)
        plt.xlabel('Date')
        plt.ylabel('USD')
        plt.title(stock + ' price')
        plt.legend()
        plt.show()
    
    def measurement(self, yhat,plot=True):
        '''this method works'''
        if self.splited == -1:
            #this is the case when our class does not memorize the train x datetime
            raise ValueError("You didn\'t split data")
        ####@@@@@@@@@@@@@@@@@@@
        time_index = self.data.iloc[self.test_idx,]['Date']
        yhat = pd.DataFrame({'yhat':yhat,'Date':time_index})
        stock=self.target
        # then the plot function
        if plot and self.random==-1:

            plt.figure(figsize=(14, 5), dpi=100)
            plt.plot(self.data['Date'], self.data[stock], label="Real Stock Price")
            plt.plot(yhat['Date'],yhat['yhat'],label='Predicted Stock Price')
            plt.legend(['True Price',"Predicted Price"])
            plt.title(stock+' price')
            plt.show()
            
        if plot and self.random==1:

            plt.figure(figsize=(14, 5), dpi=100)
            #plt.plot(self.data.iloc[self.test_idx,].sort_values(by=['Date'])['Date'], #self.data.iloc[self.test_idx,].sort_values(by=['Date'])[stock], label="Real Stock Price")
            plt.plot(self.data['Date'], self.data[stock], label="Real Stock Price")
            plt.plot(yhat['Date'],
                        yhat['yhat'],'o',color='red',markersize=2.5)
            plt.title(stock + ' price')
            plt.legend(["True Price","Predicted Price"])
            plt.title(stock+' price')      
            plt.show()
        
        # we calculate the mean squared error
        mse = mean_squared_error(self.data.iloc[self.test_idx,][stock],yhat['yhat'])
        print("The mean squared error is "+str(mse))
        return {'mse':mse}
    
    
    
    def helper_rolling_window(self, price, window):
        '''this method is a helper function, it will create the similar time window output'''
        X = []
        y = []
        for i in range(window, price.shape[0]):
            #print(i)
            X.append(price[i-window:i])
            y.append(price[i])
        X, y = np.array(X), np.array(y)
        
        return X,y

    def data_split_window(self, split_ratio=0.8, wind_size=255, plot=True, random=False, accuracy = False):
        '''this function will use the provided window size to split the data into training and testing dataset
        the plot function is used to plot the split place
        and it will return all of them'''
        
        if self.read == -1:
            raise ValueError("You didn't import_data") 
    
        stock=self.target
        data = self.data[stock]
        X,y = self.helper_rolling_window(data, wind_size) 
        size = int(X.shape[0]*split_ratio) #this is the number of samples in training data.
        
        if random==True:
            self.random=1
            np.random.seed(self.seed)
            indices = np.random.permutation(X.shape[0])
            training_idx, test_idx = indices[:size], indices[size:]
            X_train, X_test = X[training_idx,:], X[test_idx,:]
            y_train, y_test = y[training_idx], y[test_idx]
            
            index_train = self.data.iloc[training_idx+wind_size,]['Date']
            index_test = self.data.iloc[test_idx+wind_size,]['Date']
            #store test index
            self.test_idx = test_idx+wind_size
            
            fin_X_train = pd.DataFrame(X_train,index=index_train)
            fin_X_test = pd.DataFrame(X_test,index = index_test)
            fin_y_train = pd.DataFrame(y_train,index=index_train)
            fin_y_test = pd.DataFrame(y_test,index=index_test)
            self.splited = 1
            if plot:
                plt.figure(figsize=(14, 5), dpi=100)
                plt.plot(self.data['Date'], self.data[stock], label=stock)
                plt.plot(self.data.iloc[test_idx+wind_size,:]['Date'],
                         self.data.iloc[test_idx+wind_size,:][stock],'o',color='red',markersize=2.5)
                plt.xlabel('Date')
                plt.ylabel('USD')
                plt.title(stock + ' price')
                plt.legend(["train data","test data"])
                plt.show()
            return (fin_X_train,fin_y_train),(fin_X_test,fin_y_test)
        
        else: #when random is not true, we split the data sequentially  
            self.random=-1
            X_train = X[:size,]
            X_test = X[size:,]
            
            y_train = y[:size]
            y_test = y[size:]
            
            index_train = self.data.iloc[wind_size:size+wind_size,]['Date']
            index_test = self.data.iloc[size+wind_size:,]['Date']
            
            #store test index
            self.test_idx = list(range(size+wind_size,self.data.shape[0]))
                             
            fin_X_train = pd.DataFrame(X_train,index=index_train)
            fin_X_test = pd.DataFrame(X_test,index = index_test)
            fin_y_train = pd.DataFrame(y_train,index=index_train)
            fin_y_test = pd.DataFrame(y_test,index=index_test)
            self.splited = 1
            split_date = self.data.iloc[size+wind_size]['Date'] #this is the date that separate train and test data
            
            #then plot the split line
            if plot:
                plt.figure(figsize=(14, 5), dpi=100)
                plt.plot(self.data['Date'], self.data[stock], label=stock)
                plt.vlines(split_date, 0, 270, linestyles='--', colors='red', label='Train/Test data cut-off')
                plt.xlabel('Date')
                plt.ylabel('USD')
                plt.title(stock + ' price')
                plt.legend()
                plt.show()
            
            return (fin_X_train,fin_y_train),(fin_X_test,fin_y_test)
        
    def generate_technical_indicators(self, plot=True, last_days=400):
        stock=self.target
        
        dataset = self.data.copy()
        
        dataset = dataset[['Date',stock]]
        # Create 7 and 21 days Moving Average
        dataset['ma7'] = dataset[stock].rolling(window=7).mean()
        dataset['ma21'] = dataset[stock].rolling(window=21).mean()
    
        # Create MACD the difference of 26 and 12 exponential moving average
        dataset['26ema'] = dataset[stock].ewm(span=26).mean()
        dataset['12ema'] = dataset[stock].ewm(span=12).mean()
        dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

        # Create Bollinger Bands this is similar to the confidence interval
        dataset['20sd'] = dataset[stock].rolling(window=20, center=False).std()
        dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
        dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
        # Create Exponential moving average
        dataset['ema'] = dataset[stock].ewm(com=0.5).mean()
        
        # Create Momentum
        #p0, p1, p2, p3 = calc_momentum(dataset, k=30)
        # we will use more other momentum method laters
        dataset['momentum'] = dataset[stock].diff()
        #for the log momentum, I have made some slight change: first take log and then take difference
        #https://www.tradingview.com/script/UxIcFNAz-Log-Momentum/
        dataset['log_momentum'] = np.log(dataset[stock]).diff()

        if plot:
            self.plot_technical_indicators(dataset,stock,last_days)
        dataset=dataset.set_index('Date')
        dataset = dataset.drop(columns=stock)
        return dataset
    
    def plot_technical_indicators(self, dataset, stock, last_days):
        '''this is actually a helper function to plot technical indicators.'''
        plt.figure(figsize=(16, 10), dpi=100)
        
        shape_0 = dataset.shape[0]

        dataset = dataset.iloc[-last_days:, :]
        date = pd.to_datetime(dataset['Date'])

        #xmacd_ = shape_0-last_days  #the maximun of the plot
        
        # Plot first subplot
        plt.subplot(2, 1, 1)
        plt.plot(dataset['Date'],dataset['ma7'],label='MA 7', color='g',linestyle='--')
        plt.plot(dataset['Date'],dataset[stock],label='Closing Price', color='b')
        plt.plot(dataset['Date'],dataset['ma21'],label='MA 21', color='r',linestyle='--')
        plt.plot(dataset['Date'],dataset['upper_band'],label='Upper Band', color='c')
        plt.plot(dataset['Date'],dataset['lower_band'],label='Lower Band', color='c')
        plt.fill_between(dataset['Date'], dataset['lower_band'], dataset['upper_band'], alpha=0.35)
        plt.title('Technical indicators for '+stock +' - last {} days.'.format(last_days))
        plt.ylabel('USD')
        plt.legend()

        # Plot second subplot
        plt.subplot(2, 1, 2)
        plt.title('MACD')
        plt.plot(dataset['Date'],dataset['MACD'],label='MACD', linestyle='-.')
        plt.hlines(15, dataset['Date'].iloc[0,], dataset['Date'].iloc[-1,], colors='g', linestyles='--') #specifies to xmin and xmax
        plt.hlines(-15, dataset['Date'].iloc[0,], dataset['Date'].iloc[-1,], colors='g', linestyles='--')
        plt.plot(dataset['Date'],dataset['momentum'],label='Momentum', color='b',linestyle='-')

        plt.legend()
        plt.show()
    
            
    def generate_momentum(self, wind_size=30, plot=True, last_days=500):
        stock=self.target
        x = np.log(self.data[stock])
        v = x.diff()
        m =  self.data['Volume']
        p0 = v.rolling(window=wind_size, center=False).sum()
        mv = m*v
        p1 = mv.rolling(window=wind_size, center=False).sum()
        p2 = p1/(m.rolling(window=wind_size, center=False).sum())
        p3 = v.rolling(window=wind_size, center=False).mean()/v.rolling(window=wind_size, center=False).std()
        
        result = pd.DataFrame( {'p0':p0.values,'p1':p1.values,'p2':p2.values,'p3':p3.values},index=self.data.Date )
        if plot:
            self.helper_plot_momentum(result,last_days)
        return result
    
    def helper_plot_momentum(self, momentum_data,last_days):
        p0=momentum_data['p0'][-last_days:]
        p1=momentum_data['p1'][-last_days:]
        p2=momentum_data['p2'][-last_days:]
        p3=momentum_data['p3'][-last_days:]
        f, ax1 = plt.subplots(figsize=(15,7))
        ax1.plot(p0)
        ax2 = ax1.twinx()
        ax2.plot(p1,'r')
        ax1.plot(p2)
        ax1.plot(p3)
        ax1.set_title('Momentum of '+self.target)
        ax1.legend(['p(0)', 'p(2)', 'p(3)'], bbox_to_anchor=(1.25, 1))
        ax2.legend(['p(1)'], bbox_to_anchor=(1.25, .75))

        plt.show()
    
    def generate_fft(self, num_components, plot=True):
        
        stock = self.target
        
        close_fft = np.fft.fft(np.asarray(self.data[stock].tolist()))
        fft_df = pd.DataFrame({'fft':close_fft})
        fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
        
        dataset = self.data[['Date',stock]]
        dataset = dataset.set_index('Date')
        fft_list = np.asarray(fft_df['fft'].tolist())
        for num_ in num_components:
            fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
            #plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
            strFFT = 'FFT' + str(num_)
            dataset[strFFT] = np.fft.ifft(fft_list_m10).real
            
        if plot:
            self.helper_fft_plot(num_components,fft_df)
        dataset = dataset.drop(columns=stock)
        return dataset
            
    def helper_fft_plot(self, num_components, fft_df):
        stock=self.target
        
        plt.figure(figsize=(14, 7), dpi=100)
        fft_list = np.asarray(fft_df['fft'].tolist())
        for num_ in num_components:
            fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
            plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
            
        plt.plot(self.data[stock],label='Real')
        plt.xlabel('Days')
        plt.ylabel('USD')
        plt.title(stock +  ' stock prices & Fourier transforms')
        plt.legend()
        plt.show()
        
    def scale(self,X_train,y_train,X_test,y_test):
        X_scaler = MinMaxScaler(feature_range=(0,1))
        X_scaler.fit(X_train)
        X_train_fin = pd.DataFrame(X_scaler.transform(X_train),columns=X_train.columns,index=X_train.index)
        X_test_fin = pd.DataFrame(X_scaler.transform(X_test),columns=X_test.columns,index=X_test.index)
        self.X_scaler=X_scaler
        #then transform the y_test
        y_scaler = MinMaxScaler(feature_range=(0,1))
        y_scaler.fit(y_train)
        y_train_fin = pd.DataFrame(y_scaler.transform(y_train),columns=y_train.columns,index=y_train.index)
        y_test_fin = pd.DataFrame(y_scaler.transform(y_test),columns=y_test.columns,index=y_test.index)
        self.y_scaler = y_scaler
        
        return (X_train_fin,y_train_fin),(X_test_fin,y_test_fin)
    
    def inverse_scale(self,X_train,y_train,X_test,y_test):
        X_train_fin = pd.DataFrame(self.X_scaler.inverse_transform(X_train),columns=X_train.columns,index=X_train.index)
        X_test_fin = pd.DataFrame(self.X_scaler.inverse_transform(X_test),columns=X_test.columns,index=X_test.index)
        y_train_fin = pd.DataFrame(self.y_scaler.inverse_transform(y_train),columns=y_train.columns,index=y_train.index)
        y_test_fin = pd.DataFrame(self.y_scaler.inverse_transform(y_test),columns=y_test.columns,index=y_test.index)
        
        return (X_train_fin,y_train_fin),(X_test_fin,y_test_fin)
        
    #arima result
    def generate_arima(self, order):
        '''order is the d p q paramter to fit an arima model
        to generate feature by arima, we purpose a bidirectional arima
        we generate the second half of data by the correct direction, and we generate the first half of the data by the reverse direction'''
        stock = self.target
    
        #let's get the first half done
        half_time = int(self.data.shape[0]*0.5)  #this is half data
        train = self.data[stock].values[:half_time].tolist()  #hence train and test will be just numpy array
        #print(len(train))
        test = self.data[stock].values[half_time:].tolist()
        #generate our prediction
        prediction = list()
    
        history = copy.deepcopy(train)
        for t in range(len(test)):
            model = smt.ARIMA(history, order=order)
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = float(output[0])
            prediction.append(yhat)
            obs = test[t]
            history.append(obs)
        #print(len(train))
        print("first half implementation finished")
        
        
        #now let's do the second half partition
        (test,train) = (train,test) #switch order
        test.reverse()
        train.reverse()
        prediction2 = list()
    
        history = copy.deepcopy(train)
        for t in range(len(test)):
            model = smt.ARIMA(history, order=order)
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = float(output[0])
            prediction2.append(yhat)
            obs = test[t]
            history.append(obs)
        prediction2.reverse()
        test.reverse()
        pred = prediction2+ prediction
        
        ypred_df = pd.DataFrame(pred,index=self.data['Date'],columns=['arima'])
        
        return(ypred_df)
    
    def generate_logreturn(self):
        stock = self.target
        dataset=self.data.copy()
        log_return = np.log(self.data[stock]/self.data[stock].shift(1))  #this will output nan but we will also include index 
        dataset['log_return']=log_return
        dataset = dataset.set_index('Date')
        dataset = dataset.drop(columns=[stock,'Volume'])
        return dataset
        