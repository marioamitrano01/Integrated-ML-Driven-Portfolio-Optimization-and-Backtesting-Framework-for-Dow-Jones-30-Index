import os
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam



def log_function_call(func):
    
    def wrapper(*args, **kwargs):
        print(f"[*] Calling function: {func.__name__}")
        print(f"    - args: {args[1:] if len(args) > 1 else None}")
        print(f"    - kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"[*] Execution completed: {func.__name__}\n")
        return result
    return wrapper



class AttentionLayer(tf.keras.layers.Layer):


    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    


    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    


    def call(self, inputs):
      


        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b) 
        attention_weights = tf.nn.softmax(score, axis=1)  
        context_vector = attention_weights * inputs 
        context_vector = tf.reduce_sum(context_vector, axis=1) 
        return context_vector




class DowConfig:


    TICKERS = {
        "3M": "MMM",
        "American Express": "AXP",
        "Amgen": "AMGN",
        "Apple": "AAPL",
        "Boeing": "BA",
        "Caterpillar": "CAT",
        "Chevron": "CVX",
        "Cisco": "CSCO",
        "Coca-Cola": "KO",
        "Dow": "DOW",
        "Goldman Sachs": "GS",
        "Home Depot": "HD",
        "Honeywell": "HON",
        "IBM": "IBM",
        "Intel": "INTC",
        "Johnson & Johnson": "JNJ",
        "JPMorgan Chase": "JPM",
        "McDonald's": "MCD",
        "Merck": "MRK",
        "Microsoft": "MSFT",
        "Nike": "NKE",
        "Pfizer": "PFE",
        "Procter & Gamble": "PG",
        "Salesforce": "CRM",
        "Travelers": "TRV",
        "UnitedHealth": "UNH",
        "Verizon": "VZ",
        "Visa": "V",
        "Walgreens Boots Alliance": "WBA",
        "Walmart": "WMT"
    }


    OPTIMIZATION_START_DATE = "2020-01-01"
    OPTIMIZATION_END_DATE = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    BACKTEST_START_DATE = "2020-01-01"
    BACKTEST_END_DATE = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    

    INITIAL_CAPITAL = 100000



class DowDataLoader:


    def __init__(self, start_date: str, end_date: str):


        self.start_date = start_date
        self.end_date = end_date



    @log_function_call
    def load_data(self, ticker: str) -> pd.DataFrame:
       


        df = yf.download(ticker, start=self.start_date, end=self.end_date)
        df.reset_index(inplace=True)
        if "Adj Close" in df.columns:
            df = df[['Date', 'Adj Close']].rename(columns={'Adj Close': 'Close'})
        else:
            df = df[['Date', 'Close']]
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @log_function_call
    def load_all(self) -> dict:
        
        data = {}
        for company, ticker in DowConfig.TICKERS.items():
            data[company] = self.load_data(ticker)
        return data



class ReturnsCalculator:


    @staticmethod
    def compute_log_return(df, shift_days: int = 1):
        return np.log(df["Close"] / df["Close"].shift(shift_days))
    


    @log_function_call
    def add_returns_and_volatility(self, df: pd.DataFrame,
                                   window_daily=2, window_weekly=5, window_monthly=21) -> pd.DataFrame:
        df["log_return"] = self.compute_log_return(df)
        df.dropna(inplace=True)
        df["vol_daily"] = df["log_return"].rolling(window=window_daily).std()
        df["vol_weekly"] = df["log_return"].rolling(window=window_weekly).std()
        df["vol_monthly"] = df["log_return"].rolling(window=window_monthly).std()
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df



class PortfolioOptimizer:
    def __init__(self, num_portfolios=5000, risk_aversion=1.0):
        self.num_portfolios = num_portfolios
        self.risk_aversion = risk_aversion
    


    @log_function_call
    def optimize(self, combined_df: pd.DataFrame):


       
        returns = np.log(combined_df / combined_df.shift(1)).dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(mean_returns)
        
        results = np.zeros((3, self.num_portfolios))
        weights_record = []
        


        for i in range(self.num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            objective = portfolio_return - self.risk_aversion * portfolio_std
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std
            results[2, i] = objective
        
        max_objective_idx = np.argmax(results[2])
        optimal_weights = weights_record[max_objective_idx]
        portfolios_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Objective"])
        for i, company in enumerate(combined_df.columns):
            portfolios_df[str(company) + "_weight"] = [w[i] for w in weights_record]
        
        return optimal_weights, portfolios_df



class MLPortfolioOptimizer:


    def __init__(self, window_size=30, epochs=200, batch_size=32, learning_rate=0.001, risk_aversion=1.0):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.risk_aversion = risk_aversion
        self.model = None



    def build_model(self, num_assets):
        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(self.window_size, num_assets)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(AttentionLayer())
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_assets, activation='softmax'))
        
        optimizer = Adam(learning_rate=self.learning_rate)
       


        def objective_loss(y_true, y_pred):
            portfolio_returns = tf.reduce_sum(y_true * y_pred, axis=1)
            mean_ret = tf.reduce_mean(portfolio_returns)
            std_ret = tf.math.reduce_std(portfolio_returns)
            objective = mean_ret - self.risk_aversion * std_ret
            epsilon = 1e-6
            entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + epsilon), axis=1)
            lambda_entropy = 0.1
            num_assets_tf = tf.cast(tf.shape(y_pred)[1], tf.float32)
            uniform = tf.ones_like(y_pred) / num_assets_tf
            lambda_uniform = 0.05
            uniform_penalty = tf.reduce_mean(tf.square(y_pred - uniform))
            return - (objective + lambda_entropy * tf.reduce_mean(entropy)) + lambda_uniform * uniform_penalty
        


        model.compile(optimizer=optimizer, loss=objective_loss)
        self.model = model



    def prepare_data(self, combined_df: pd.DataFrame):
        


        returns_df = np.log(combined_df / combined_df.shift(1)).dropna()
        data = returns_df.values  
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i-self.window_size:i])
            y.append(data[i])
        X = np.array(X)
        y = np.array(y)
        return X, y



    @log_function_call
    def train(self, X, y):
        num_assets = X.shape[2]
        if self.model is None:
            self.build_model(num_assets)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)



    @log_function_call
    def optimize(self, X):
        
        weights_pred = self.model.predict(X)
        optimal_weights = np.mean(weights_pred, axis=0)
        return optimal_weights



class Backtester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital



    @log_function_call
    def run_backtest(self, combined_df: pd.DataFrame, weights: np.array,
                     start_date: str, end_date: str) -> pd.DataFrame:
       
        backtest_df = combined_df.loc[start_date:end_date]
        returns = np.log(backtest_df / backtest_df.shift(1)).dropna()
        portfolio_returns = returns.dot(weights)
        portfolio_cum_value = self.initial_capital * np.exp(portfolio_returns.cumsum())
        
        vol_daily = portfolio_returns.rolling(window=2).std()
        vol_weekly = portfolio_returns.rolling(window=5).std()
        vol_monthly = portfolio_returns.rolling(window=21).std()
        
        sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
        
        running_max = portfolio_cum_value.cummax()
        drawdown = (portfolio_cum_value - running_max) / running_max
        max_drawdown = drawdown.min()
        
        backtest_results = pd.DataFrame({
            "Portfolio": portfolio_cum_value,
            "Portfolio_Daily_Return": portfolio_returns,
            "Vol_Daily": vol_daily,
            "Vol_Weekly": vol_weekly,
            "Vol_Monthly": vol_monthly
        })
        backtest_results["Sharpe_Ratio"] = sharpe_ratio
        backtest_results["Max_Drawdown"] = max_drawdown
        backtest_results.index.name = "Date"
        return backtest_results



class DowPortfolioApplication:


    def __init__(self):
        self.loader_opt = DowDataLoader(DowConfig.OPTIMIZATION_START_DATE, DowConfig.OPTIMIZATION_END_DATE)
        self.loader_backtest = DowDataLoader(DowConfig.BACKTEST_START_DATE, DowConfig.BACKTEST_END_DATE)
        self.portfolio_optimizer = PortfolioOptimizer(num_portfolios=5000, risk_aversion=1.0)
        self.ml_optimizer = MLPortfolioOptimizer(window_size=30, epochs=200, batch_size=32,
                                                  learning_rate=0.001, risk_aversion=1.0)
        self.combined_df_opt = None
        self.combined_df_backtest = None



    @log_function_call
    def run(self):


        data_opt = self.loader_opt.load_all()
        combined_df_opt = None
        for company, df in data_opt.items():
            df = df[["Date", "Close"]].set_index("Date")
            if combined_df_opt is None:
                combined_df_opt = df.rename(columns={"Close": company})
            else:
                combined_df_opt = combined_df_opt.join(df.rename(columns={"Close": company}), how="inner")
        combined_df_opt = combined_df_opt.dropna()
        self.combined_df_opt = combined_df_opt



        data_back = self.loader_backtest.load_all()
        combined_df_back = None
        for company, df in data_back.items():
            df = df[["Date", "Close"]].set_index("Date")
            if combined_df_back is None:
                combined_df_back = df.rename(columns={"Close": company})
            else:
                combined_df_back = combined_df_back.join(df.rename(columns={"Close": company}), how="inner")


        combined_df_back = combined_df_back.dropna()
        self.combined_df_backtest = combined_df_back



        optimal_mc_weights, portfolios_df = self.portfolio_optimizer.optimize(combined_df_opt)
        print("Optimal Portfolio Weights (Monte Carlo Simulation):")
        for company, weight in zip(combined_df_opt.columns, optimal_mc_weights):
            print(f"  {company}: {weight:.2%}")




        X, y = self.ml_optimizer.prepare_data(combined_df_opt)
        split_idx = int(0.8 * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]



        self.ml_optimizer.train(X_train, y_train)




        ml_optimal_weights = self.ml_optimizer.optimize(X_test)
        print("\nOptimal Portfolio Weights (ML Optimization):")
        for company, weight in zip(combined_df_opt.columns, ml_optimal_weights):
            print(f"  {company}: {weight:.2%}")

        backtester = Backtester(DowConfig.INITIAL_CAPITAL)
        backtest_results = backtester.run_backtest(self.combined_df_backtest, ml_optimal_weights,
                                                     DowConfig.BACKTEST_START_DATE, DowConfig.BACKTEST_END_DATE)




        dow_index = yf.download("^DJI", start=DowConfig.BACKTEST_START_DATE, end=DowConfig.BACKTEST_END_DATE)
        dow_index.reset_index(inplace=True)
        if "Adj Close" in dow_index.columns:
            dow_index = dow_index[['Date', 'Adj Close']].rename(columns={'Adj Close': 'Dow_Close'})


        else:


            dow_index = dow_index[['Date', 'Close']].rename(columns={'Close': 'Dow_Close'})
        dow_index['Date'] = pd.to_datetime(dow_index['Date'])
        dow_index.sort_values('Date', inplace=True)
        dow_index.set_index('Date', inplace=True)
        dow_index["Dow_Return"] = np.log(dow_index["Dow_Close"] / dow_index["Dow_Close"].shift(1))
        dow_index.dropna(inplace=True)
        dow_index["Dow_Cum"] = DowConfig.INITIAL_CAPITAL * np.exp(dow_index["Dow_Return"].cumsum())
        dow_index.reset_index(inplace=True)

        
        if isinstance(dow_index.columns, pd.MultiIndex):
            dow_index.columns = dow_index.columns.get_level_values(0)



        backtest_results = backtest_results.reset_index()
        merged = pd.merge(backtest_results, dow_index, on="Date", how="inner")

        


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged["Date"], y=merged["Portfolio"], mode="lines",
                                 name="ML Portfolio Value"))
        fig.add_trace(go.Scatter(x=merged["Date"], y=merged["Dow_Cum"], mode="lines",
                                 name="Dow Jones 30 Index Value"))
        fig.add_trace(go.Scatter(x=merged["Date"], y=merged["Vol_Daily"], mode="lines",
                                 name="Daily Volatility", yaxis="y2"))
        fig.add_trace(go.Scatter(x=merged["Date"], y=merged["Vol_Weekly"], mode="lines",
                                 name="Weekly Volatility", yaxis="y2"))
        fig.add_trace(go.Scatter(x=merged["Date"], y=merged["Vol_Monthly"], mode="lines",
                                 name="Monthly Volatility", yaxis="y2"))



        fig.update_layout(
            title="Backtest of Enhanced ML Optimized Dow Jones 30 Portfolio vs. Dow Jones 30 Index",
            xaxis_title="Date",
            yaxis=dict(title="Portfolio Value (USD)", side="left"),
            yaxis2=dict(title="Volatility (Std. Dev of Log Returns)",
                        overlaying="y", side="right", showgrid=False),
            legend=dict(x=0.01, y=0.99),
            template="plotly_dark"
        )
        fig.show()



@log_function_call
def main():
    app = DowPortfolioApplication()
    app.run()




if __name__ == "__main__":
    main()
