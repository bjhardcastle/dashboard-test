import logging
from typing import Literal

import npc_lims
import panel as pn
import plotly.express as px
import polars as pl

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@pn.cache
def get_all_unit_locations_df() -> pl.DataFrame:
    paths = tuple(
        npc_lims.get_cache_path(
            component, 
            version='v0.0.231', 
            consolidated=True,
        )
        for component in ('units', 'session', 'performance')
    )
    logger.info(f"Reading dataframes from {paths}")
    units, session, performance = [pl.read_parquet(p) for p in paths]
    location_df = (
        session.filter(
            pl.col('keywords').list.contains('templeton').not_()
        )
        .join(
            performance.filter(
                pl.col('same_modal_dprime') > 1.0,
                pl.col('cross_modal_dprime') > 1.0,
            ).group_by(
                pl.col('session_id')).agg(
                [
                    (pl.col('block_index').count() > 3).alias('pass'),
                ],  
            ).filter('pass').drop('pass'),
            on='session_id',
        )
        .join(
            units.filter(
                pl.col('isi_violations_count') < 0.5,
                pl.col('amplitude_cutoff') < 0.1,
                pl.col('presence_ratio') > 0.95,
            ),
            on='session_id',
        )
        .group_by([
            pl.col('session_id'),
            pl.col('location'),
        ]).agg(
            [
                pl.col('location').count().alias('location_count'),
                pl.col('structure').first(),
                pl.col('subject_id').first(),
                pl.col('date').first(),
                pl.col('unit_id').explode().alias('unit_ids'),
            ]
        )
    )
    logger.info(f"Found unit locations after filtering:{' '.join(sorted(s for s in location_df['location'].unique() if s is not None))}")
    return location_df

@pn.cache
def get_location_query_df(
    search_term: str, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
) -> pl.DataFrame:
    
    if not case_sensitive:
        search_term = search_term.lower()
        location_col = pl.col('location').str.to_lowercase()
    else:
        location_col = pl.col('location')
    location_expr = getattr(location_col.str, search_type)(search_term)
    
    df = (
        get_all_unit_locations_df()
        .filter(location_expr)
        .sort('date', "location")
    )
    logger.info(f"Found {len(df)} units: location.{search_type}({search_term}, {case_sensitive=})")
    return df

def plot_unit_locations_bar(
    search_term: str | None, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    group_by: Literal['session_id', 'subject_id'] = 'subject_id',
) -> pn.pane.Plotly | None:
    
    if not search_term:
        return pn.pane.Plotly(None)

    df = get_location_query_df(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive)
    
    s = df['structure'].unique()
    if len(s) > 1:
        logger.warning(f"Multiple structures found: {list(s)}")
        structure = s.to_list()
    elif len(s) == 0:
        logger.warning(f"No structures found")
        structure = []
    else:
        structure = s[0]
    
    fig = px.bar(
        df.cast({"subject_id": str}),       # make subject column str so we don't have big gaps on x-axis
        x=group_by, 
        y="location_count", 
        color="location", 
        category_orders={"location": df['location'].unique().sort()},   # sort entries in legend
        labels={'location_count': 'units'}, 
        hover_data="session_id", 
        title=f"breakdown of good units in {structure} in good sessions (total = {len(df)} units)", 
    ) 
    fig.update_layout(
        autosize=True,
        width=None,
    )
    return pn.pane.Plotly(fig)

def plot_co_recorded_structures_bar(
    search_term: str | None, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
):
    query_df = get_location_query_df(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive)
    df = (
        get_all_unit_locations_df()
        .filter(
            pl.col('session_id').is_in(query_df['session_id'].unique())
        )
        .explode('unit_ids')
        .rename({'unit_ids': 'unit_id'})
        .filter(
            pl.col('unit_id').is_in(query_df['unit_ids'].explode().unique()).not_()
        )
    )

    df = (
        df
        .group_by(pl.col('structure', 'session_id')).agg([
            pl.col('unit_id').count().alias('unit_count'), 
        ])
        .join(
            df.group_by(pl.col('structure')).agg(pl.col('structure').count().alias('total_structure_count')),
            on='structure',
        )
        .group_by('structure').agg(
            [   
                pl.col('session_id'),
                pl.col('unit_count'),
                pl.col('total_structure_count').first(),
            ]
        )
        .top_k(k=15, by='total_structure_count')
        .sort('total_structure_count', descending=True) 
        .explode('session_id', 'unit_count') 
    )
    
    fig = px.bar(
        df,
        x="structure", 
        y="unit_count",
        hover_data="session_id",
        labels={'unit_count': 'units'}, 
        title=f"structures co-recorded in sessions with {search_term}", 
        barmode='group',
    ) 
    fig.update_layout(
        autosize=True,
        width=None,
    )
    return pn.pane.Plotly(fig)

# add a dropdown selector for the search type and a text input for the search term
search_type_input = pn.widgets.Select(name='Search type', options=['starts_with', 'contains'], value='starts_with')
search_term_input = pn.widgets.TextInput(name='Search location', value='MOs')
search_case_sensitive_input = pn.widgets.Checkbox(name='Case sensitive', value=False)
group_by_input = pn.widgets.Select(name='Group by', options=['session_id', 'subject_id'], value='subject_id')

# column of plots
bound_plot_unit_locations_bar = pn.bind(plot_unit_locations_bar, search_term=search_term_input, search_type=search_type_input, case_sensitive=search_case_sensitive_input, group_by=group_by_input)
bound_plot_co_recorded_structures_bar = pn.bind(plot_co_recorded_structures_bar, search_term=search_term_input, search_type=search_type_input, case_sensitive=search_case_sensitive_input)
plots = pn.Column(bound_plot_unit_locations_bar, bound_plot_co_recorded_structures_bar)

pn.template.MaterialTemplate(
    site="Dynamic Routing",
    title="units by location",
    sidebar=[group_by_input, search_type_input, search_term_input, search_case_sensitive_input],
    main=[plots],
).servable()

