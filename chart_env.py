import pandas as pd
import numpy as np
import time

import os
from PIL import Image
import cv2


import gym
from gym import spaces




class ChartEnv(gym.Env):

        
    SIZE = 1000
    displayresolution = (1920, 1080)
    
    CURRENTPRICE = 0
    CANDLE = 0
    PRICE = np.array([])
    OLD_PRICEMOVES = {0:CURRENTPRICE}

    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    

    BALANCE = 1000
    EQUITY = 1000
    FEE = 0

    OPEN_POSITIONS = pd.DataFrame(columns=['index', 'direction', 'price'])
    LONGS = 0
    SHORTS = 0

    TOTALPROFIT = 0
    LONGPROFIT = 0
    SHORTPROFIT = 0

    done = False

    def __init__(self, symbol, df, point):
        self.SYMBOL = symbol
        self.POINT = point
        self.CHART = pd.DataFrame(df)
        self.CHART = self.ProcessTicks(self.CHART)
        self.PRICE = np.append(self.PRICE, self.SIZE/2)


    def ProcessTicks(self, ticks):
        self.CHART = pd.DataFrame(ticks)
        self.CHART = self.CHART[['time', 'bid', 'ask']]
        self.CHART['time']=pd.to_datetime(self.CHART['time'], unit='s').dt.floor('Min')
        self.CHART.drop_duplicates(subset=['time'], inplace=True)
        self.CHART['price'] = self.CHART[['bid', 'ask']].mean(axis=1)
        self.CHART.drop(['ask', 'bid','time'], axis=1, inplace=True)
        self.CHART.dropna(axis=0, inplace=True)
        return self.CHART



    def Buy(self):
        self.LONGS +=1  
        self.OPEN_POSITIONS.loc[self.LONGS] = self.LONGS, 0, self.PRICE[-1]
        self.BALANCE -= self.FEE
    
    def Sell(self):
        self.SHORTS +=1    
        self.OPEN_POSITIONS.loc[self.CANDLE] = self.SHORTS, 1, self.PRICE[-1]
        self.BALANCE -= self.FEE



    def step(self):

        self.CANDLE += 1

        if 2 <= self.CANDLE < self.CHART.shape[0]-1:
            self.CURRENTPRICE = self.CHART.iloc[self.CANDLE-1]
            previousprice = self.CHART.iloc[self.CANDLE-2]
            currentmove = (self.CURRENTPRICE  - previousprice ) //self.POINT
            self.OLD_PRICEMOVES[self.CANDLE] = currentmove
            self.PRICE = np.append(self.PRICE, self.PRICE[self.CANDLE - 1] + currentmove)


        else:
            self.OLD_PRICEMOVES[self.CANDLE] = 0
            self.PRICE = np.append(self.PRICE, self.PRICE[self.CANDLE-1])





        self.TOTALPROFIT = 0
        self.LONGPROFIT = 0
        self.SHORTPROFIT = 0

        for position in self.OPEN_POSITIONS.itertuples():
            if position is not None:
                if int(position[2]) == 0:
                    self.TOTALPROFIT += (self.PRICE[self.CANDLE] - position[3]) # * Lot Sise
                    self.LONGPROFIT += (self.PRICE[self.CANDLE] - position[3])

                elif int(position[2]) == 1:
                    self.TOTALPROFIT += (position[3] - self.PRICE[self.CANDLE])
                    self.SHORTPROFIT += (position[3] - self.PRICE[self.CANDLE])
        
        self.EQUITY = self.BALANCE + self.TOTALPROFIT




    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        start = self.PRICE[0]
        for candle, price in self.OLD_PRICEMOVES.items():
            start += price
            env[int(start), int( candle)] = (255, 165, 0) 

        for o in self.OPEN_POSITIONS.itertuples():
                if o is not None:
                    if int(o[2]) == 0:
                        env[int(o[3]), :]    = (0, 255, 0) # green

                    elif int(o[2]) == 1:
                        env[int(o[3]), :]    = (0, 0, 255) # Red


        img = Image.fromarray(env, 'RGB')  
        return img



    def render(self):
        img = self.get_image()
        img = img.resize(self.displayresolution)
        img = np.array(img)
        img = cv2.flip(img, 0) 

 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f'Price :  {self.PRICE[self.CANDLE] }', (10,20), font, fontScale=0.5,color=(255,255,255),thickness=1,lineType=cv2.LINE_AA)
        cv2.putText(img, f'Long :  {self.LONGS} - P/L : {self.LONGPROFIT} $', (10,40), font, fontScale=0.5,color=(255,255,255),thickness=1,lineType=cv2.LINE_AA)
        cv2.putText(img, f'Short : {self.SHORTS} - P/L : {self.SHORTPROFIT} $',(10,60), font, fontScale=0.5,color=(255,255,255),thickness=1,lineType=cv2.LINE_AA)
        cv2.putText(img, f'Total Profit : {self.TOTALPROFIT} $',(10,100), font, fontScale=0.6,color=(255,255,255),thickness=1,lineType=cv2.LINE_AA)
        cv2.putText(img, f'Account Balance : {self.BALANCE} $',(10,120), font, fontScale=0.6,color=(255,255,255),thickness=1,lineType=cv2.LINE_AA)
        cv2.putText(img, f'Account Equity : {self.EQUITY} $',(10,140), font, fontScale=0.6,color=(255,255,255),thickness=1,lineType=cv2.LINE_AA)

        cv2.imshow("image", img) 

        cv2.waitKey(1)

df = pd.read_csv('Tick_Data\\BTCUSD\\1-1_ticks.csv')
pip = 10

env = ChartEnv("BTCUSD", df, pip)

while True:
    env.step()
    env.render()

    t_end = time.time() + 1.2
    k = -1
    while time.time() < t_end:
        if k == -1:
            k = cv2.waitKey(50)
        else:
            continue

    if k == ord('b') :
        env.Buy()
    elif k == ord('s'):
        env.Sell()
    elif k == ord('q'):
        break