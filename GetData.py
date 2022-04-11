import MetaTrader5 as mt5   # pip install MetaTrader5
import pandas as pd
import datetime as dt
import pytz




def Connect():
    if not mt5.initialize(
        path='',                    # link to the MT5 terminal.exe  
        login=0,                    # MT5 account number
        password="",                # MT5 account password
        server="",                  # MT5 account server      
    ): 
        print("initialize() failed")
        return False
    else:
        return True

def Disconnect():
    mt5.shutdown()


def Get_Data_By_Ticks(symbol, start_time, end_time):
    if Connect() == True:
        data = pd.DataFrame()
        data = mt5.copy_ticks_range(symbol, start_time, end_time, mt5.COPY_TICKS_ALL)
        if data is None:
            print('Error Gitting Data:\n', mt5.last_error())
            Disconnect()
            return None
        else:
            print("Ticks received:",len(data))
        Disconnect()
        return(data)
    else:
        print("Error Coccecting to MT5")
        return None 

def Save_Data(name, data):
    df = pd.DataFrame(data)
    df.to_csv(f'{name}.csv',sep=',')



symbol = 'BTCUSD'
date_from = dt.datetime(year=2022, month=1, day=1, tzinfo=pytz.timezone("Etc/UTC"))
date_to = dt.datetime(year=2022, month=2, day=1, tzinfo=pytz.timezone("Etc/UTC"))


filename = f'Tick_Data/{symbol}/{date_from.month}-{date_to.month}_ticks'
data = Get_Data_By_Ticks(symbol, date_from, date_to)

if data is not None and len(data) > 0:
    print(f'Saving Data {symbol} _ {date_from} : {date_to}')
    Save_Data(filename, data)