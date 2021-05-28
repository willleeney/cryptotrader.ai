import pandas as pd
import plotly.graph_objects as go

def plot_df(df, filename, quote_symbol, base_symbol):
    
    candlestick = go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )

    fig = go.Figure(data=[candlestick])

    fig.update_layout(
        width=800, height=600,
        title=filename,
        yaxis_title=f'{quote_symbol}/{base_symbol}'
    )

    fig.show()

