import pandas as pd
from tensortrade.feed.core import Stream, DataFeed

def rsi(price: Stream[float], period: float) -> Stream[float]:
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100 * (1 - (1 + rs) ** -1)

def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal



def create_basic_features(df: pd.DataFrame):

    df['change'] = df['open'] - df['close']
    df['body'] = abs(df['change'])

    df['abs'] =  df['change'].apply(lambda x: 1 if x > 0 else -1 )
    #df['abs'] = df.apply(lambda x: calc_abs(df['change']), axis=1)


    #df['abs']  = [1 if row > 0 else -1 for row in df['change']]

    df['agreewick'] = [row.high - row.close if row.abs == 1 else row.close - row.low for row in df.itertuples()]
    df['diswick'] = [row.open - row.low if row.abs == 1 else row.high - row.open for row in df.itertuples()]

    df[df['agreewick'] == 0] = 0.01
    df[df['diswick'] == 0] = 0.01

    df['agreebodywick'] = [row.body/ row.agreewick for row in df.itertuples()]
    df['disbodywick'] = [row.body / row.diswick for row in df.itertuples()]

    df = df.drop(['agreewick', 'diswick'], axis=1, inplace=False)


    return df

