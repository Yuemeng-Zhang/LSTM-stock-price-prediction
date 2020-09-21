import pandas as pd
from datetime import datetime
import time
from dateutil.relativedelta import relativedelta
import requests
from io import StringIO

import requests, re, json

from concurrent import futures

class downloader:
    def __init__(self,directory):
        '''the initialize method need a working directory to download these files'''
        if not isinstance(directory,str):
            raise ValueError("You should input a working directory")
        self.dir = directory
        
    def make_url(self, ticker_symbol, unix_start, unix_end, daily):
        '''this function is used to generate yahoo link to fetch stock price history
        it is also called in pull_historical_data method'''
        if daily:
            link = 'https://finance.yahoo.com/quote/' + ticker_symbol +'/history?period1='+str(unix_start)+'&period2='+str(unix_end)+'&filter=history&interval=1d&frequency=1d'
        else:
            link ='https://finance.yahoo.com/quote/' + ticker_symbol +'/history?period1='+str(unix_start)+'&period2='+str(unix_end)+'&filter=history&interval=1mo&frequency=1mo'
        return link
    
    def calc_start_end_time(self, num_yrs=5):
        '''this is a helper function, it will calculate start time and years by a given number of years
           we typically want to use it when the start or end is not specified'''
        end_date = str(datetime.now()).split()[0]
        start_date = str(datetime.now() + relativedelta(years = - num_yrs)).split()[0]
        print('start date: ', start_date)
        print('end date: ', end_date)
        start = datetime(int(start_date.split('-')[0]), int(start_date.split('-')[1]), int(start_date.split('-')[2]))
        end = datetime(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))
        unix_start = int(time.mktime(start.timetuple()))
        day_end = end.replace(hour=23, minute=59, second=59)
        unix_end = int(time.mktime(day_end.timetuple()))
        return unix_start, unix_end
    
    
    def pull_stock(self, stocklist, daily, start_date=-1, end_date=-1,  delay = 0.5, is_save = False):
        '''this method is nothing more than a parllel implementation of pulling stock price'''
        output_path= self.dir   
        
        if start_date==-1 and end_date==-1:
            #if start_date and end_date is not specified, we will use a default 5 year 
            unix_start,unix_end = self.calc_start_end_time()
        else:
            #otherwise we will use the input date
            start=datetime(int(start_date.split('-')[0]), int(start_date.split('-')[1]), int(start_date.split('-')[2]))
            end=datetime(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))
            unix_start=int(time.mktime(start.timetuple()))
            day_end=end.replace(hour=23, minute=59, second=59)
            unix_end=int(time.mktime(day_end.timetuple()))
            
        #here we need also to check the type of stocklist, if it's string, change it to list
        if isinstance(stocklist, str):
            stocklist=[stocklist]
        
        
        #we will store a stock_df in the class, because it's easier to concatenate it in the parllel process
        self.stock_df = pd.DataFrame()
        #the same reasons for bad_names and good_names
        self.bad_names = []
        self.good_names = []
        
        #first identify the maximun number of workers.
        max_workers = 40
        workers = min(max_workers, len(stocklist))
        
        #we then use the method helper_pull_historical_data to run these tasks parllelly
        with futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future = executor.map(self.helper_pull_stock, stocklist, [unix_start]*len(stocklist), [unix_end]*len(stocklist), [daily]*len(stocklist), [is_save]*len(stocklist),[delay]*len(stocklist))
        
        #we then formalize the final dataset: stock_df
        self.stock_df.columns = self.good_names
        self.stock_df.index = pd.to_datetime(self.stock_df.index)
        stock_df = self.stock_df.sort_index(ascending = True)
        stock_df['Date'] = stock_df.index
        stock_df.reset_index(drop=True, inplace=True)
        return stock_df

    def helper_pull_stock(self, stock, unix_start, unix_end, daily, is_save=False, delay=0.5):
        '''this is a for one-instance pull stock price history, we will need it in the parallel algorithm to extract list of stocks. You should never call it directly because it needs variable of good_names and others which should be assigned first in pull_stock method'''
        output_path=self.dir
        try:
            '''this part is the main part to pull historical data'''
            r = requests.get(self.make_url(stock, unix_start, unix_end, daily))
            ptrn = r'root\.App\.main = (.*?);\n}\(this\)\);'
            tmpTxt = re.search(ptrn, r.text, re.DOTALL)
            df = pd.DataFrame()
            if tmpTxt is not None:
                txt = tmpTxt.group(1)
                jsn = json.loads(txt)
                df = pd.DataFrame(jsn['context']['dispatcher']['stores']['HistoricalPriceStore']['prices'])
                df.insert(0, 'symbol', stock)
                df['date'] = pd.to_datetime(df['date'], unit='s').dt.date
                df = df.dropna(subset=['close'])
                df = df[['symbol', 'date', 'high', 'low', 'open', 'close', 'volume', 'adjclose']]
                df = df.sort_values(by='date',ascending=True)
                if is_save:   
                    filename = output_path + "/" + stock + "_price.csv"
                    df.to_csv(filename)
            df.set_index('date', inplace = True)
            self.stock_df = pd.concat([self.stock_df, df['adjclose']], sort = True, axis = 1)
            self.good_names.append(stock)
        except:
            self.bad_names.append(stock)
            print('bad stock: ', stock)
            
        time.sleep(delay)
        
    def pull_company_table(self, stocklist, delay=0.5, frequency ="annual", is_save=True):
        '''this is a parllel implementation of pulling company tables. 
        It will download all balance sheet, income statement, and cashflow for each company'''
        if isinstance(stocklist, str):
            stocklist=[stocklist]
        #first identify the maximun number of workers.
        max_workers = 40
        workers = min(max_workers, len(stocklist))
        
        #we then use the method helper_pull_historical_data to run these tasks parllelly
        with futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future = executor.map(self.helper_pull_company_table, stocklist, [frequency]*len(stocklist), [is_save]*len(stocklist))
        print("download finshed")
        
    
    def helper_pull_company_table(self, stock, frequency_str, is_save=True):
        '''this is a one-instance helper function of pull-company data 
        it will save all three tables of the companies
        Also this method will automatically save the tables'''
        
        output_path=self.dir
        
        if frequency_str == 'annual':
            do_ratio=True
            frequency = '12'
        elif frequency_str == 'quarterly':
            do_ratio=False
            frequency = '3'
        report_lis = ['is','bs','cf']
        reportfullname_lis=['income-statement','balance-sheet','cashflow-statement']
        
        for i in range(len(report_lis)):
            report = report_lis[i]
            reportfullname = reportfullname_lis[i]
            url = 'http://financials.morningstar.com/ajax/ReportProcess4CSV.html?t=' + stock + '&reportType=' + report + '&period=' + frequency + '&dataType=A&order=asc&columnYear=5&number=3'
            hdr = {
                'Referer': 'http://financials.morningstar.com/' + reportfullname + '/' + report + '.html?t=' + stock + '&region=usa&culture=en-US'}
            req = requests.get(url, headers=hdr)
            df = pd.DataFrame()
            if len(req.text) > 0:
                data = StringIO(req.text)
                df = pd.read_csv(data, skiprows=1, index_col=0)
    
                if is_save: 
                    filename = output_path + "/" + stock + "_" + report + "_" + frequency_str + ".csv"
                    df.to_csv(filename)
        
        #if frequency is anuual we should also download the ratio table.
        if do_ratio:
            url = 'http://financials.morningstar.com/finan/ajax/exportKR2CSV.html?&callback=?&t='+ stock+'&region=usa&culture=en-US&cur=&order=asc'
            hdr = {'Referer': 'http://financials.morningstar.com/ratios/r.html?t='+stock+'&region=USA&culture=en-US'}
            req = requests.get(url, headers=hdr)
            df = pd.DataFrame()
            if len(req.text) > 0:
                data = StringIO(req.text)
                df = pd.read_csv(data, skiprows=2, index_col=0)
                if is_save:
                    filename = output_path + "/" + stock + "_ratio.csv"
                    #here the split criteria is by tab, we should check if this is correct
                    df.to_csv(filename,sep='\t')

        