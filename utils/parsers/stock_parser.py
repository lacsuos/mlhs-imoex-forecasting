import requests
import re
import pandas as pd 
from io import StringIO, BytesIO
from datetime import datetime, timedelta


class StockMarketParser:
    def __init__(self):
        self.imoex_url = 'https://export.finam.ru/export9.out?apply=0&p=8&e=.csv&dtf=1&tmf=1&MSOR=0&mstimever=on&sep=3&sep2=1&datf=1&at=1&from={df}.{mf}.{yf}&to={dt}.{mt}.{yt}&em=420450&code=IMOEX&f=IMOEX_150412_200411&cn=IMOEX&market=undefined&yf={yf}&yt={yt}&df={df}&dt={dt}&mf={mf}&mt={mt}'
        self.brent_url = 'https://export.finam.ru/export9.out?apply=0&p=8&e=.csv&dtf=1&tmf=1&MSOR=0&mstimever=on&sep=3&sep2=1&datf=1&at=1&from={df}.{mf}.{yf}&to={dt}.{mt}.{yt}&em=4185457&code=BZ&f=BZ_150412_200411&cn=BZ&market=undefined&yf={yf}&yt={yt}&df={df}&dt={dt}&mf={mf}&mt={mt}'
        self.imoex_data = None
        self.brent_data = None
        self.cbr_data = None
        self.usdrub_data = None

    def __split_period(self, start_date, end_date):
        periods = []
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=4*365), end_date)
            periods.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)
        return periods
    

    def _finam_parser(self, start_date, end_date, url):

        all_data = []      
        for period_start, period_end in self.__split_period(start_date, end_date):
            # Формируем URL для каждого периода
            period_url = url.format(
                yf=period_start.year,
                mf=period_start.month-1,  # Finam использует нумерацию месяцев с 0
                df=period_start.day,
                yt=period_end.year,
                mt=period_end.month-1,
                dt=period_end.day
            )
            
            # Делаем запрос
            response = requests.get(period_url, stream=True)
            if response.status_code != 200:
                return response.status_code
            
            # Парсим CSV и добавляем к общим данным
            csv_data = StringIO(response.text)
            df_part = pd.read_csv(csv_data, delimiter=';')
            all_data.append(df_part)
        
        # Объединяем все части
        if all_data:
            full_df = pd.concat(all_data, ignore_index=True)
            full_df = full_df.rename({col: re.sub(r'\W+', '', col).lower() for col in full_df.columns}, axis=1)
            full_df['date'] = full_df['date'].astype(int).astype(str).apply(pd.to_datetime)
            full_df = full_df.sort_values('date')
            return full_df
        else:
            return None

    
    def _cbr_parser(self, start_date, end_date):
        url = 'https://www.cbr.ru/Queries/UniDbQuery/DownloadExcel/132934?Posted=True&FromDate={mf}%2F{df}%2F{yf}&ToDate={mt}%2F{dt}%2F{yt}'    
        response = requests.get(url.format(
                yf=start_date.year,
                mf=start_date.month,  # Finam использует нумерацию месяцев с 0
                df=start_date.day,
                yt=end_date.year,
                mt=end_date.month,
                dt=end_date.day
            ), stream=True)
        if response.status_code==200:
            csv_data = BytesIO(response.content)
            cbr_df = pd.read_excel(csv_data, dtype={'Дата': 'str'})
            cbr_df['Дата'] = cbr_df['Дата'].apply(lambda x: pd.to_datetime(str(x), format='%m.%Y'))
            cbr_df.rename(
                {'Дата': 'date', 
                'Ключевая ставка, % годовых': 'key_rate', 
                'Инфляция, % г/г': 'cpi', 
                'Цель по инфляции': 'cpi_target'}, 
                axis=1,
                inplace=True)
            cbr_df.sort_values('date', inplace=True)
        return cbr_df
    
    def _usdrub_parser(self, start_date, end_date):
        start_date = start_date.strftime('%d/%m/%Y')
        end_date = end_date.strftime('%d/%m/%Y')
        url = f'https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1={start_date}&date_req2={end_date}&VAL_NM_RQ=R01235'
        print(url)
        response = requests.get(url, stream=True)
        if response.status_code==200:
            print(response.text)
            xml_data = StringIO(response.text)
            usdrub = pd.read_xml(xml_data, encoding='windows-1251', xpath="//Record")
            usdrub.rename({'Date': 'date', 'Value': 'usdrub'}, axis=1, inplace=True)
            usdrub.drop(['Id', 'Nominal', 'VunitRate'], inplace=True, axis=1)
            usdrub['usdrub'] = usdrub['usdrub'].str.replace(',', '.').astype(float)
            usdrub['date'] = usdrub['date'].apply(pd.to_datetime, dayfirst=True)
        return usdrub 
    
    def get_data(self, start_date, end_date):
        self.imoex_data = self._finam_parser(start_date, end_date, self.imoex_url)
        self.brent_data = self._finam_parser(start_date, end_date, self.brent_url)[['date', 'close']].rename({'close': 'brent'}, axis=1)
        self.cbr_data = self._cbr_parser(start_date, end_date)
        self.usdrub_data = self._usdrub_parser(start_date, end_date)
        print(self.imoex_data.date.dtype, self.brent_data.date.dtype, self.usdrub_data.date.dtype)
        prepared_df = pd.merge(self.imoex_data, self.brent_data, on='date')
        prepared_df = pd.merge_asof(prepared_df, self.usdrub_data, on='date')
        prepared_df = pd.merge_asof(prepared_df, self.cbr_data[['key_rate', 'cpi', 'date']], on='date')
        return prepared_df