import pandas as pd


def create_basic_features(df: pd.DataFrame):

    df['change'] = df['open'] - df['close']
    df['body'] = abs(df['change'])

    df['abs'] =  df['change'].apply(lambda x: 1 if x > 0 else -1 )
    #df['abs'] = df.apply(lambda x: calc_abs(df['change']), axis=1)


    #df['abs']  = [1 if row > 0 else -1 for row in df['change']]

    df['agreebodywick'] = [row.body/(row.high - row.close) if row.abs == 1 \
                               else row.body/(row.close - row.low) \
                               for row in df.itertuples()]

    df['disbodywick'] = [row.body / (row.open - row.low) if row.abs == 1 \
                           else row.body / (row.high - row.open) \
                           for row in df.itertuples()]

    return df

