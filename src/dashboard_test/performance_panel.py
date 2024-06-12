from typing import Iterable, Literal

import holoviews as hv
import pandas as pd
import panel as pn
import plotly.express as px
import polars as pl

pd.options.plotting.backend = "plotly"

pn.extension('plotly', 'tabulator')


@pn.cache
def get_data(table_name: Literal['performance', 'session', 'subject']) -> pl.LazyFrame:
    """
    >>> data = get_data('performance')
    >>> data.head(2).collect()
    shape: (2, 15)
    ┌────────────┬─────────────┬─────────────┬───────────────────┬───┬────────────┬────────────┬─────────────────────┬─────┐
    │ start_time ┆ stop_time   ┆ block_index ┆ rewarded_modality ┆ … ┆ date       ┆ subject_id ┆ session_id          ┆ id  │
    │ ---        ┆ ---         ┆ ---         ┆ ---               ┆   ┆ ---        ┆ ---        ┆ ---                 ┆ --- │
    │ f64        ┆ f64         ┆ i64         ┆ str               ┆   ┆ date       ┆ i64        ┆ str                 ┆ i64 │
    ╞════════════╪═════════════╪═════════════╪═══════════════════╪═══╪════════════╪════════════╪═════════════════════╪═════╡
    │ 3.438755   ┆ 947.832445  ┆ 0           ┆ vis               ┆ … ┆ 2022-03-29 ┆ 614910     ┆ 614910_2022-03-29_0 ┆ 0   │
    │ 7.487278   ┆ 3608.823576 ┆ 0           ┆ vis               ┆ … ┆ 2022-03-30 ┆ 614910     ┆ 614910_2022-03-30_0 ┆ 0   │
    └────────────┴─────────────┴─────────────┴───────────────────┴───┴────────────┴────────────┴─────────────────────┴─────┘
    >>> data.columns
    ['start_time', 'stop_time', 'block_index', 'rewarded_modality', 'cross_modal_dprime', 'signed_cross_modal_dprime', 'same_modal_dprime', 'nonrewarded_modal_dprime', 'vis_intra_dprime', 'aud_intra_dprime', 'session_idx', 'date', 'subject_id', 'session_id', 'id']    """
    return pl.scan_parquet(f's3://aind-scratch-data/ben.hardcastle/cache/nwb_components/v0.0.209/consolidated/{table_name}.parquet')

@pn.cache
def get_session_mean_dprime(performance_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Each session has 1 or more context blocks. This function returns the
    average of the aud and vis intra dprime values across the blocks in each session.
        
    >>> get_session_mean_dprime().head(2)
    shape: (2, 7)
    ┌─────────────────────┬────────────┬────────────┬────────────────┬─────────────┬────────────┬──────────────────────────┐
    │ session_id          ┆ subject_id ┆ date       ┆ session_number ┆ mean_dprime ┆ num_blocks ┆ days_since_first_session │
    │ ---                 ┆ ---        ┆ ---        ┆ ---            ┆ ---         ┆ ---        ┆ ---                      │
    │ str                 ┆ i64        ┆ date       ┆ u32            ┆ f64         ┆ u32        ┆ i64                      │
    ╞═════════════════════╪════════════╪════════════╪════════════════╪═════════════╪════════════╪══════════════════════════╡
    │ 614910_2022-03-30_0 ┆ 614910     ┆ 2022-03-30 ┆ 2              ┆ 0.29937     ┆ 1          ┆ 0                        │
    │ 614910_2022-03-31_0 ┆ 614910     ┆ 2022-03-31 ┆ 3              ┆ 0.577372    ┆ 1          ┆ 1                        │
    └─────────────────────┴────────────┴────────────┴────────────────┴─────────────┴────────────┴──────────────────────────┘
    """
    df = performance_df or get_data('performance')
    
    # add column with mean of aud and vis intra dprime
    df = df.with_columns(
        df.select(
            pl.col('vis_intra_dprime'), 
            pl.col('aud_intra_dprime')
            ).collect()
        .mean_horizontal(ignore_nulls=True)
        .alias('mean_dprime')
        )
    
    # get session number before dropping any rows
    df = df.with_columns(
        (pl.col('stop_time') - pl.col('start_time')).alias('block_duration'),
        pl.col('session_id').rank('dense').over('subject_id').alias('session_number'),
        )
    df = df.drop_nulls('mean_dprime')

    df = df.group_by(
        pl.col('session_id')).agg([
            pl.col('subject_id').first(),
            pl.col('date').first(), 
            pl.col('session_number').first(),
            pl.col('mean_dprime').mean().alias('mean_dprime'),
            pl.col('block_index').count().alias('num_blocks'),
        ],                      
    )
    # add a column for the number of days since the first session for each
    # subject_id
    df = (
        df
        .with_columns(
            (pl.col('date') - pl.col('date').min().over('subject_id')).dt.total_days().alias('days_since_first_session'),
        )
    )
    
    df = df.sort(['subject_id', 'date'])
    return df.collect()


def parse_session_ids(subject_ids: str | Iterable[int] | None) -> tuple[int, ...]:
    """
    >>> parse_session_ids('1, 2 3  4')
    (1, 2, 3, 4)
    """
    if not subject_ids:
        return ()
    if isinstance(subject_ids, str):
        subject_ids = subject_ids.replace(',', ' ').replace('  ', ' ').strip()
        subject_ids = tuple(int(x) for x in subject_ids.split(' '))
    else:
        subject_ids = tuple(subject_ids)
        assert all(isinstance(x, int) for x in subject_ids)
    return subject_ids

def plot_session_mean_dprime_over_time(subject_ids: str | Iterable[int] | None = None) -> hv.Curve:
    """
    """
    df = get_session_mean_dprime().with_columns(pl.col('date').cast(str))
    if subject_ids:
        df = df.filter(pl.col('subject_id').is_in(parse_session_ids(subject_ids)))
    fig = px.line(df, x="days_since_first_session", y="mean_dprime", color="subject_id", line_group="subject_id", hover_data=["date", "session_number", "num_blocks",])
    # fig.update_traces(mode="lines+markers", marker=dict(size=10), line=dict(width=4))
    fig.layout.autosize = True
    fig.update_layout(showlegend=False)
  

    return pn.pane.Plotly(fig)
    # return f.hvplot.line(x='days_since_first_session', y='mean_dprime', hover_cols=['date', 'num_blocks'], groupby=['subject_id']).overlay()


subject_id_input_widget = pn.widgets.TextInput(
    name='Subject ID Input', 
    placeholder='Enter int IDs (space or comma sep)...',
)
subjects_table_widget = df_widget = pn.widgets.Tabulator(get_data('session').collect().to_pandas())
bound_plot = pn.bind(
    plot_session_mean_dprime_over_time, 
    subject_ids=subject_id_input_widget, 
)

# widgets = pn.Row(subject_id_input_widget, sizing_mode="fixed", width=300)

pn.Column(subject_id_input_widget, bound_plot, subjects_table_widget)


pn.template.MaterialTemplate(
site="Panel",
title="Getting Started App",
# sidebar=[subject_id_input_widget],
main=[subject_id_input_widget, bound_plot, subjects_table_widget]
).servable()

if __name__ == '__main__':
    # print(','.join(get_session_mean_dprime().get_column('subject_id').cast(str).unique()))
    import doctest
import pandas as pd
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
