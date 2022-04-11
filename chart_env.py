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


    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        start = self.PRICE[0]
        for candle, price in self.OLD_PRICEMOVES.items():
            start += price
            env[int(start), int( candle)] = (255, 165, 0) 

        img = Image.fromarray(env, 'RGB')  
        return img



    def render(self):
        img = self.get_image()
        img = img.resize(self.displayresolution)
        img = np.array(img)
        img = cv2.flip(img, 0) 

        cv2.imshow("image", img) 

        cv2.waitKey(1)




df = pd.read_csv('Tick_Data\\BTCUSD\\1-2_ticks.csv')
pip = 10
env = ChartEnv("BTC", df, pip)

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
        
    if k == ord('q'):
        break

cv2.destroyAllWindows()