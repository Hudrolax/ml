import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Iterable


def add_scatter(fig, line: list | np.ndarray | pd.Series, color: str | None = None) -> go.Figure:
    kwargs = dict()
    if color is not None:
        kwargs['line'] = dict(color=color)
    
    if isinstance(line, pd.Series) or isinstance(line, np.ndarray) and len(line.shape) == 1:
        kwargs['x'] = np.arange(0, len(line), 1)

    fig.add_trace(go.Scatter(
        y=line,
        mode='lines',
        **kwargs
    ))
    return fig


def render_lines(lines: list | np.ndarray | pd.Series | pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not isinstance(lines, Iterable):
        raise TypeError('Lines must be iterable object.')
    else:
        if len(lines) == 0:
            raise ValueError('Lines is empty.')

    if isinstance(lines, list):
        for line in lines:
            fig = add_scatter(fig, line)
    else:
        fig = add_scatter(fig, lines)

    # style   
    fig.update_layout(
        {
            'plot_bgcolor': "#151822",
            'paper_bgcolor': "#151822",
        },
        font=dict(color='#dedddc'),
        showlegend=False,
        margin=dict(b=20, t=0, l=0, r=40),
    )
    fig.update_yaxes(
        showgrid=True,
        zeroline=False,
        showticklabels=True,
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        showline=True,
        spikedash='dot',
        side='right',
        spikethickness=1,
    )

    return fig
