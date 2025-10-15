import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K
import mlflow
import mlflow.keras


def sharpe_ratio_loss(y_true, y_pred):
    """
    Custom loss function that maximizes the Sharpe Ratio.
    y_true: actual forward returns.
    y_pred: predicted signals/weights (magnitude indicates conviction).
    """
    # Calculate portfolio returns
    portfolio_returns = y_true * y_pred

    # Calculate mean and standard deviation of returns
    mean_returns = K.mean(portfolio_returns)
    std_returns = K.std(portfolio_returns)

    # Calculate Sharpe Ratio (assuming risk-free rate = 0), add epsilon for stability
    sharpe_ratio = mean_returns / (std_returns + K.epsilon())

    # Minimize the negative Sharpe Ratio
    return -sharpe_ratio


def build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.2, use_sharpe_loss=True):
    """
    Builds a stacked LSTM model for time series prediction.
    """
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    # Output a signal between -1 and 1
    model.add(Dense(1, activation='tanh'))

    loss = sharpe_ratio_loss if use_sharpe_loss else 'mean_squared_error'
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def train_and_log_model(X_train, y_train, X_val, y_val, model_params, experiment_name="Quant_Alpha_LSTM"):
    """
    Trains the LSTM model and logs the experiment with MLflow.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(model_params)

        model = build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            lstm_units=model_params.get('lstm_units', 64),
            dropout_rate=model_params.get('dropout_rate', 0.2),
            use_sharpe_loss=model_params.get('use_sharpe_loss', True)
        )

        history = model.fit(
            X_train, y_train,
            epochs=model_params.get('epochs', 100),
            batch_size=model_params.get('batch_size', 32),
            validation_data=(X_val, y_val),
            verbose=1
        )

        # Log metrics
        val_loss = history.history['val_loss'][-1]
        mlflow.log_metric("val_loss", val_loss)
        if model_params.get('use_sharpe_loss', True):
            mlflow.log_metric("val_sharpe_ratio", -val_loss)

        # Log the model artifact
        mlflow.keras.log_model(model, "model")
        return model
