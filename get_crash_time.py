# -*- coding: utf-8 -*-

import tushare as ts
import pandas as pd
import numpy as np
import ssl
import csv
import matplotlib.pyplot as plt
import pickle as pk

stock_list=[]
res_string = []
event_dict = dict()
crash_sample = dict()
#crash_matrix = np.zeros()

def get_stock_code():
    """初始化股票代码list"""
    with open('../data/stock_list.csv') as fin:
        for line in fin.readlines():
            line = line.replace(".SH","")
            line = line.replace(".SZ","")
            line = line.strip()
            stock_list.append(line)
            

def get_stock_history_price():
    """获取股票的历史价格数据"""
    ssl._create_default_https_context = ssl._create_unverified_context
    for item in stock_list:
        code = item.strip()
        print (code)
        # df = ts.get_hist_data(code,start='2015-01-01')
        df = ts.get_k_data(code,start='2015-01-01')
        if df is None: continue
        #直接保存
        df.to_csv('../data/' + code + '.csv')

    df = ts.get_k_data('sz',start='2015-01-01')
    df.to_csv('../data/sz.csv')
    df = ts.get_k_data('sh',start='2015-01-01')
    df.to_csv('../data/sh.csv')

def get_stock_price(code):
    code = code.replace(".SH",'')
    code = code.replace(".SZ",'')
    data = pd.read_csv('../data/'+code+'.csv')
    return data

def check_if_crash():
    """
    暴跌定义为：
        0.某个交易日内跌幅位于5%-10%之间（新发股可能跌破10%）
        1.3个交易日内跌幅达到30%及以上
        2.7个工作日内跌幅达到50%及以上
    """
    stock_index = 1;
    for item in stock_list:
        #crash_record['stock_code'].append(item)
        #crash_record['crash_date'].append([])
        data = get_stock_price(item)
        if len(data)==0: continue
        date = data['date']
        open_price = data['open']
        close_price = data['close']
        length = len(close_price)
        res = ''
        #0
        for i in range(length):
            percentage_0 = (open_price[i] - close_price[i])/open_price[i]
            if percentage_0 > 0.05 and percentage_0 <= 0.1:
                res = item + ',' + date[i] + ',' + str(open_price[i]) + ',' + str(close_price[i]) + ',' + str(percentage_0) + ',' + "某个交易日跌幅超过5%"
                # print(res)
                res_string.append(res)

        #1
        for i in range(length-2):
            percentage_1 = (open_price[i] - close_price[i+2])/open_price[i]
            if percentage_1 > 0.30:
                res = item + ',' + date[i] + ',' + str(open_price[i]) + ',' + str(close_price[i+2]) +',' + str(percentage_1) + ','+ "2个交易日内跌幅达到30%及以上"
                res_string.append(res)

               
        #2
        for i in range(length-7):
            percentage_2 = (open_price[i] - close_price[i+7])/open_price[i]
            if percentage_2 > 0.50:
                res = item + ',' + date[i] + ',' + str(open_price[i]) + ',' + str(close_price[i+7]) +',' + str(percentage_2) + ',' + "7个交易日内跌幅达到50%及以上"
    
    
        stock_index += 1 
        
    # save crash time for each stock 
def write_to_crash_time():    
    fileheader = ['stock_code','crash_date']
    with open('../data/res/crash_time.csv','w+',newline ='') as csvfile:
        writer = csv.DictWriter(csvfile, fileheader)
        writer.writeheader()
        for i in range(len(crash_record)):
            writer.writerow(crash_record[i+1])
    #print(len(crash_record))    
        
def write_to_csv():
    head = "stock_code,date,open,close,percentage,detail"
    with open('../data/res/crash_up_to_date.csv','w+',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(head.split(','))
        for row in res_string:
            writer.writerow(row.split(','))

# def norm_date(text_date):
    # # transform the text_date  to  numpy form
    # text_date = text_date.strip()
    # data = get_stock_price(stock_list[0])
    # if text_date not in data['date']:
        # print('date name is error.')
    # date = data['date'].index(text_date)
    # return date
        
def visualize_crash_time():
    data = pd.read_csv('../data/res/crash_up_to_date.csv',encoding = "gb2312",dtype={'stock_code':str})
    data_sh =  pd.read_csv('../data/sh.csv')
    plt.figure()
    plt.subplot(3,1,1)
    data.groupby('date')['stock_code'].count().plot()
    plt.subplot(3,1,2)
    data_sh['volume'].plot()
    plt.subplot(3,1,3)
    data_sh['close'].plot()
    plt.figure()
    #data['stock_code'].value_counts().sort_values(ascending = False).plot(logx ='True', logy = 'True')
    data['stock_code'].value_counts().hist(bins = 30).plot()
    
    fig, ax = plt.subplots()
    #data['stock_code'].value_counts().sort_values(ascending = False).plot(logx ='True', logy = 'True')
    data['date'].value_counts().hist(ax=ax)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
        
def get_stock_info():
    df = ts.get_stock_basics()
    df.to_csv('../data/stock_info.csv')

def get_type(stock_code):
    # 输入股票代码，获取公司所属的行业类型
    # df = ts.get_industry_classified()
    # df.to_csv('../data/type_info.csv')
    df = pd.read_csv('../data/type_info.csv',dtype={'code':str})
    name =  df[df['code'] == stock_code]['c_name']
    return name
    
def uni_date_form(x):
    s = x.Date[4:14].strip()
    # s.replace(("['",""))
    # s.replace("']","")   
    str_list = s.split(' ')    
    return pd.to_datetime(str_list[0])

def uni_code_form(x):
    s = x['EnterpriseCode'].replace("['","")
    s = s.replace("']","")
    s = s.replace(".SZ","")
    l = s.split(', ')
    return l
    
def data_clean():   
    #clean the data set of News_result.csv
    df = pd.read_csv('../data/news/News_result.csv') 
    data= df[df.EnterpriseType!= '其他'].copy() # 去除“其他”类型的新闻
    data = data.dropna(axis=0,how='all')
    data['Date'] = data.apply(uni_date_form,axis = 1) #整理日期格式，只保留 年月日 格式
    data = data[data.EnterpriseCode!='[nan]']
    data['EnterpriseCode'] = data.apply(uni_code_form,axis = 1) #整理 股票代码 格式，去除空记录 
    
    df_news = pd.DataFrame(columns=['date', 'stock_list','event_type'])
    df_news['date'] = data['Date']  
    df_news['event_type'] = data['EnterpriseType']
    df_news['stock_list'] = data['EnterpriseCode']
    df_news = df_news.reset_index(drop = True)
    #df_news.to_csv('../data/news/data_news.csv')
    df_news.to_pickle('../data/news/data_news.pkl')

    
def get_event_feature():
    df = pd.read_pickle('../data/news/data_news.pkl')
    type_dict = {'监管处罚': np.array([0,0,0,0,0,1]), \
    '投资并购': np.array([0,0,0,0,1,0]) , '股东减值': np.array([0,0,0,1,0,0]), \
    '定向增发': np.array([0,0,1,0,0,0]), '业绩大增': np.array([0,1,0,0,0,0]), '股权激励': np.array([1,0,0,0,0,0])}
    df.event_type = df.event_type.map(type_dict)
    for indexs, items in df.iterrows():
        event_dict[indexs] = dict()
        for code in items['stock_list']:
            if code != '':
                event_dict[indexs]['code'] = code
                event_dict[indexs]['date'] = items['date']
                event_dict[indexs]['event_type'] =items['event_type']
            
    data_event = pd.DataFrame.from_dict(event_dict,orient='index')
    data_event.to_pickle('../data/news/data_event.pkl') # 每条数据存入 data_event.pkl，获取event_feature时只要读取data_event.pkl即可

    

    
def generate_crash_samples(n_days = 15, n_event_features =6):
    """
    generate crash samples according to the file 'crash_up_to_date.csv'
    n_days is the number of days considered before crash,
    n_event_features  is the number of features of each event.
    For each crash item, the function returns a dict {‘event_features’: ,'entity_feature':}
    """
    event_matrix = np.zeros([n_days,n_event_features])
    data = pd.read_csv('../data/res/crash_up_to_date.csv',encoding = "gb2312",dtype={'stock_code':str})
    data_sh = pd.read_csv('../data/sh.csv') # date information
    data_event = pd.read_pickle('../data/news/data_event.pkl')
    #type = data['stock_code'].map(get_type)
      
    for indexs, items in data.iterrows():
        crash_sample[indexs] = dict()
        event_matrix = np.zeros([n_days,n_event_features])
        crash_date = items['date']
        start_index = data_sh[data_sh.date == crash_date].index.values[0] - n_days
        date_list = data_sh[(data_sh.index > start_index)&(data_sh.date < crash_date)].date.values #获取交易日的日期
        
        #sh_index = data_sh[data_sh.date.isin(date_list)].close # 获取上证指数
        
        data_temp = data_event[data_event.date.isin(date_list)].copy() # 获取crash之前的event 数据
        if(items['stock_code'] in data_temp.code.values):
            date_index = np.where(pd.to_datetime(date_list).values == data_temp[data_temp.code == items['stock_code']].date.values)
            event_matrix[date_index,:] = [data_temp[data_temp.code == items['stock_code']].event_type.values[0]]
        
        crash_sample[indexs]['stock_code'] = items['stock_code']
        crash_sample[indexs]['event_features'] = event_matrix
        crash_sample[indexs]['date'] = event_matrix
    df = pd.DataFrame.from_dict(crash_sample,orient='index')
    df = df.reset_index(drop = True)
    df.to_pickle('../data/res/crash_sample.pkl')
        # sh_volume = date_sh[date_list.date == item['date']].volume
        # price_hist = []
        # print(sh_index)
        # #print(date_list[date_list.date == item['date']].index)
      

    
    
    

    
    
if __name__ == '__main__':
    #get_stock_code()
    #get_stock_history_price()
    #check_if_crash()
    #write_to_csv()
    #write_to_crash_time()
    #get_stock_info()
    visualize_crash_time()
    #generate_crash_samples()
    #data_clean()
    #get_event_feature()
    #generate_crash_samples()
    print()
    