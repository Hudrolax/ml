import time
import logging
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from typing import Callable
import pandas as pd

logger = logging.getLogger(__name__)


class SocketTicker:
    def __init__(self, callback: Callable) -> None:
        self.client = UMFuturesWebsocketClient()
        self.callback = callback
        self.client.start()

    def add_socket(self, socket: str, socket_kwargs: dict) -> None:
        method = getattr(self.client, socket)
        method(
            callback=self.tick_handler,
            **socket_kwargs
        )

    def tick_handler(self, message):
        if not isinstance(message, dict):
            return

        kline = message.get('k', None)
        if kline is not None:
            # got kline
            new_kline = kline['x']
            open_time = pd.to_datetime(kline['t'], unit='ms')
            close_time = pd.to_datetime(kline['T'], unit='ms')

            self.callback(
                new_kline=new_kline,
                open_time=open_time,
                close_time=close_time,
            )

    def stop(self):
        self.client.stop()


def on_new_kline(func):
    def wrapper(*args, **kwargs):
        if args[0].tick['new_kline']:
            func(*args, **kwargs)

    return wrapper


def get_kline_ticker(symbol: str, tf: str, callback: Callable) -> SocketTicker:
    ticker = SocketTicker(callback)
    kwargs = dict(
        symbol=symbol.lower(),
        id=12,
        interval=tf,
    )
    ticker.add_socket('kline', kwargs)

    return ticker
