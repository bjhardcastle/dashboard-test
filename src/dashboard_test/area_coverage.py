import logging
from typing import Iterable, Literal, TypeVar

import matplotlib.pyplot as plt
import npc_lims
import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import polars as pl

import dashboard_test.ccf as ccf_utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@pn.cache
def get_component_lf(nwb_component: npc_lims.NWBComponentStr) -> pl.LazyFrame:
    path = npc_lims.get_cache_path(
        nwb_component, 
        version='v0.0.231', 
        consolidated=True,
    )
    logger.info(f"Reading dataframe from {path}")
    return pl.scan_parquet(path)

@pn.cache
def get_good_units_df() -> pl.DataFrame:
    good_units = (
        get_component_lf("session")
        .filter(
            pl.col('keywords').list.contains('templeton').not_()
        )
        .join(
            other=(
                get_component_lf("performance")
                .filter(
                    pl.col('same_modal_dprime') > 1.0,
                    pl.col('cross_modal_dprime') > 1.0,
                )
                .group_by(
                    pl.col('session_id')).agg(
                    [
                        (pl.col('block_index').count() > 3).alias('pass'),
                    ],  
                )
                .filter('pass')
                .drop('pass')
            ),
            on='session_id',
            how="semi", # only keep rows in left table (sessions) that have match in right table (ie pass performance)
        )
        .join(
            other=(
                get_component_lf("units")
                .filter(
                    pl.col('isi_violations_ratio') < 0.5,
                    pl.col('amplitude_cutoff') < 0.1,
                    pl.col('presence_ratio') > 0.95,
                )
            ),
            on='session_id',
        )
        .join(
            other=(
                get_component_lf('electrode_groups')
                    .rename({
                        'name': 'electrode_group_name',
                        'location': "implant_location",
                    })
                    .select('session_id', 'electrode_group_name', 'implant_location')
            ),
            on=('session_id', 'electrode_group_name'),
        )
        .with_columns((pl.col('ccf_ml') > ccf_utils.get_ml_midpoint()).alias('is_right_hemisphere'))
        .join(
            other=ccf_utils.get_ccf_structure_tree_df().lazy(),
            right_on='acronym',
            left_on='location',
        )
    ).collect()
    logger.info(f"Fetched {len(good_units)} good units")
    return good_units

DataFrameOrLazyFrame = TypeVar("DataFrameOrLazyFrame", pl.DataFrame, pl.LazyFrame)

def apply_unit_count_group_by(
    df_or_lf: DataFrameOrLazyFrame,
) -> DataFrameOrLazyFrame:
    return (
        df_or_lf
        .group_by([
            pl.col('session_id'),
            pl.col('location'),
        ])
        .agg([
            pl.col('location').count().alias('location_count'),
            pl.col('structure').first(),
            pl.col('subject_id').first(),
            pl.col('date').first(),
            pl.col('unit_id').alias('unit_ids'),
            pl.col('implant_location').first(),
            pl.col('electrode_group_name').first(),
        ])
    )
    
@pn.cache
def get_unit_location_query_df(
    filter_area: str, 
    filter_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    include_right_hemisphere: bool = False,
) -> pl.DataFrame:
    
    if not case_sensitive:
        filter_area = filter_area.lower()
        location_col = pl.col('location').str.to_lowercase()
    else:
        location_col = pl.col('location')
    location_expr = getattr(location_col.str, filter_type)(filter_area)
    
    units = (
        get_good_units_df()
        .filter(pl.col('is_right_hemisphere').eq(False) if not include_right_hemisphere else pl.lit(True))
        .lazy()
        .filter(location_expr)
        .sort('date', "location")
    ).collect()
    logger.info(f"Filtered on area, found {len(units['location'].unique())} locations across {len(units['session_id'].unique())} sessions: location.{filter_type}({filter_area}, {case_sensitive=})")
    return units

@pn.cache
def get_ccf_location_query_lf(
    filter_area: str,
    filter_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    include_right_hemisphere: bool = False,
    filter_implant_location: str | None = None,
    filter_probe_letter: str | None = None,
    whole_probe: bool = False,
) -> pl.LazyFrame:
    
    if filter_probe_letter:
        filter_probe_letter = filter_probe_letter.upper().replace('PROBE', '').replace('_', '').strip()
        electrode_group_name = f"probe{filter_probe_letter}"
    else:
        electrode_group_name = None
        
    queried_units = get_unit_location_query_df(
        filter_area=filter_area,
        filter_type=filter_type,
        case_sensitive=case_sensitive,
        include_right_hemisphere=include_right_hemisphere,
    )
    
    join_on = ['session_id', 'electrode_group_name']
    if not whole_probe:
        join_on.extend(["ccf_ap", "ccf_dv", "ccf_ml"])
    join_on = tuple(join_on)    # type: ignore

    locations = (
        get_component_lf('electrodes')
        .rename({
            'x': 'ccf_ap',
            'y': 'ccf_dv',
            'z': 'ccf_ml',
            'group_name': 'electrode_group_name',
        })
        .with_columns((pl.col('ccf_ml') > ccf_utils.get_ml_midpoint()).alias('is_right_hemisphere'))
        .filter(pl.col('is_right_hemisphere').eq(False) if not include_right_hemisphere else pl.lit(True))
        .join(
            other=(
                get_component_lf('electrode_groups')
                    .rename({
                        'name': 'electrode_group_name',
                        'location': 'implant_location',
                    })
                    .select('session_id', 'electrode_group_name', 'implant_location')
            ),
            on=('session_id', 'electrode_group_name'),
        )
        .join(
            other=queried_units.lazy(),
            on=join_on,  
            how="semi", # only keep rows in left table (electrodes) that have match in right table (ie position of queried units)
        ) 
        .filter(
            pl.col('ccf_ap') > -1,
            pl.col('ccf_dv') > -1,
            pl.col('ccf_ml') > -1,
            pl.col('implant_location').str.contains(filter_implant_location) if filter_implant_location else pl.lit(True),
            pl.col('electrode_group_name') == electrode_group_name if electrode_group_name else pl.lit(True),
        )
        .select('session_id', 'electrode_group_name', 'implant_location', 'ccf_ml', 'ccf_ap', 'ccf_dv', 'location', 'structure', 'is_right_hemisphere')
        .join(
            other=(
                ccf_utils.get_ccf_structure_tree_df().lazy()
                .select('acronym', pl.selectors.starts_with("color_"))
            ),
            right_on='acronym',
            left_on='location',
        )  
    )
    approx_n_unique_sessions = len(locations.select(pl.col('session_id').approx_n_unique()).collect())
    approx_n_unique_locations = len(locations.select(pl.col('location').approx_n_unique()).collect())
    logger.info(f"Filtered on area, found approx {approx_n_unique_locations} locations across {approx_n_unique_sessions} sessions: location.{filter_type}({filter_area}, {case_sensitive=}, {filter_implant_location=}, {whole_probe=})")
    return locations

def plot_unit_locations_bar(
    filter_area: str, 
    filter_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    include_right_hemisphere: bool = False,
    group_by: Literal['session_id', 'subject_id'] = 'subject_id',
) -> pn.pane.Plotly:
    
    if not filter_area:
        return pn.pane.Plotly(None)

    grouped_units = apply_unit_count_group_by(
        get_unit_location_query_df(filter_area=filter_area, filter_type=filter_type, case_sensitive=case_sensitive, include_right_hemisphere=include_right_hemisphere)
        )
    
    fig = px.bar(
        grouped_units.cast({"subject_id": str}),       # make subject column str so we don't have big gaps on x-axis
        x=group_by, 
        y="location_count", 
        color="location", 
        category_orders={"location": grouped_units['location'].unique().sort()},   # sort entries in legend
        labels={'location_count': 'units'}, 
        hover_data="session_id", 
        title=f"breakdown of good units ({sum(grouped_units['location_count'])}) in good sessions ({grouped_units['session_id'].drop_nulls().n_unique()})", 
    ) 
    fig.update_layout(
        autosize=True,
        width=None,
    )
    return pn.pane.Plotly(fig)

def plot_co_recorded_structures_bar(
    filter_area: str, 
    filter_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    include_right_hemisphere: bool = False,
) -> pn.pane.Plotly:
    queried_units = (
        get_unit_location_query_df(filter_area=filter_area, filter_type=filter_type, case_sensitive=case_sensitive, include_right_hemisphere=include_right_hemisphere)
        )
    other_units = (
        apply_unit_count_group_by(get_good_units_df())
        .lazy()
        .filter(
            pl.col('session_id').is_in(queried_units['session_id'])
        )
        .explode('unit_ids')
        .rename({'unit_ids': 'unit_id'})
        .filter(
            ~pl.col('unit_id').is_in(queried_units['unit_id'])
        )
        .group_by(pl.col('structure', 'session_id'))
        .agg([
            pl.col('unit_id').count().alias('unit_count'), 
        ])
    )

    other_units = (
        other_units
        .join(
            other=(
                other_units
                .group_by(pl.col('structure'))
                .agg(
                    pl.col('unit_count').sum().alias('total_structure_count')
                )
            ),            
            on='structure',
        )
        .group_by('structure')
        .agg(
            [   
                pl.col('session_id'),
                pl.col('unit_count'),
                pl.col('total_structure_count').first(),
            ]
        )
        .top_k(k=(k := 15), by='total_structure_count')
        .sort('total_structure_count', descending=True) 
        .explode('session_id', 'unit_count') 
    ).collect()
    
    fig = px.bar(
        other_units,
        x="structure", 
        y="unit_count",
        hover_data="session_id",
        labels={'unit_count': 'units'}, 
        title=f"top {k} structures co-recorded in sessions with {filter_area!r}", 
        barmode='group',
    ) 
    fig.update_layout(
        autosize=True,
        width=None,
    )
    return pn.pane.Plotly(fig)

def table_all_unit_counts(
    filter_area: str, 
    filter_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    include_right_hemisphere: bool = False,
) -> pn.pane.Plotly:
    queried_units = (
        get_unit_location_query_df(filter_area=filter_area, filter_type=filter_type, case_sensitive=case_sensitive, include_right_hemisphere=include_right_hemisphere)
        )
    all_unit_counts =(
        get_good_units_df()
        .filter(pl.col('is_right_hemisphere').eq(False) if not include_right_hemisphere else pl.lit(True))
        .group_by(pl.col('location'))
        .agg([
            pl.col('unit_id').n_unique().alias('units'), 
            pl.col('session_id').n_unique().alias('sessions'),
            pl.col('subject_id').n_unique().alias('subjects'),
            pl.col('structure').first(),
            pl.col('safe_name').first().alias('description'),
        ])
        .sort('units', descending=True)
        .with_columns(
            selected=pl.col('location').is_in(queried_units['location']),
        )
        .sort('selected', 'units', descending=True)
        .drop('selected')
        .with_row_index()
    )
    color_discrete_map = {}
    for structure in all_unit_counts['structure'].unique():
        if structure in queried_units['structure']:
            color_hex = queried_units.filter(pl.col('structure') == structure)['color_hex_triplet'][0]
        else:
            color_hex = "808080"
        color_discrete_map[structure] = f"#{color_hex}"
    
    def get_color_hex(location) -> str:
        return f"#{queried_units.filter(pl.col('location') == location)['color_hex_triplet'][0]}"
    
    def background_color_queried_locations(location_series: pd.Series) -> list[str]:
        return [
            f'background-color: {get_color_hex(location)}' 
            if location in queried_units['location']
            else ''
            for location in location_series
        ]
        
    stylesheet = """
    .tabulator-cell {
        font-size: 12px;
    }
    """
    tabulator = pn.widgets.Tabulator(
        value=all_unit_counts.drop('structure').to_pandas(),
        disabled=True,
        selectable=False,
        show_index=False,
        pagination=None,
        height=400,
        stylesheets=[stylesheet],
    )
    tabulator.style.apply(background_color_queried_locations)
    return tabulator

def table_holes_to_hit_areas(
    filter_area: str, 
    filter_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    include_right_hemisphere: bool = False,
) -> pn.pane.Plotly:
    insertions: pl.DataFrame = (
        get_unit_location_query_df(filter_area=filter_area, filter_type=filter_type, case_sensitive=case_sensitive, include_right_hemisphere=include_right_hemisphere)
        .with_columns(
            insertion_id=pl.concat_str(pl.col('session_id', 'electrode_group_name', 'implant_location'), separator='_')
        )
        # find fraction of probe + implant_location hits in this area out of all
        # placements of probes at this implant_location
        .join(
            other=(
                get_good_units_df()
                .group_by('electrode_group_name', 'implant_location')
                .agg(pl.col('session_id').n_unique().alias('insertion_count_for_probe_hole_location'))
            ),
            on=('electrode_group_name', 'implant_location'),
        )
        .group_by('electrode_group_name', 'implant_location')
        .agg([
            (pl.col('insertion_id').n_unique() / pl.col('insertion_count_for_probe_hole_location')).first().round(2).alias('rate'),
            pl.col('insertion_id').n_unique().alias('hits'),
            pl.col('insertion_count_for_probe_hole_location').first(),
        ])
        # split location into implant and hole
        .with_columns([
            (
                pl.col('implant_location')
                .str.split_exact(' ', 1)
                .struct.rename_fields(["implant", "hole"]).alias('fields')
            ),
            pl.col('electrode_group_name').str.replace('probe', '').alias('probe'),
            pl.col('insertion_count_for_probe_hole_location').alias('total'),
        ])
        .unnest('fields')
        .sort('rate', descending=True)
        .select('implant', 'hole', 'probe', 'hits', 'rate', 'total')
    )
    column_filters = {
    'implant': {'type': 'input', 'func': 'like', 'placeholder': 'Filter implant'},
    'hole': {'type': 'input', 'func': 'like', 'placeholder': 'Filter hole'},
    'probe': {'type': 'input', 'func': 'like', 'placeholder': 'Filter probe'},
    'hits': {'type': 'input', 'func': '>', 'placeholder': 'Filter hits > x'},
    'rate': {'type': 'input', 'func': '>', 'placeholder': 'Filter rate > x'},
    'total': {'type': 'input', 'func': '>', 'placeholder': 'Filter total > x'},
    }

    stylesheet = """
    .tabulator-cell {
        font-size: 12px;
    }
    """
    return pn.widgets.Tabulator(
        value=insertions.to_pandas(),
        disabled=True,
        selectable=False,
        show_index=False,
        # theme="modern",
        height=250,
        stylesheets=[stylesheet],
        header_filters=column_filters,
        layout='fit_columns', 
        # width=650,
        header_align='center', 
        text_align={'int': 'center', 'float': 'center', 'str': 'center'}, #! not working
    )

def plot_ccf_locations_2d( 
    filter_area: str, 
    filter_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    include_right_hemisphere: bool = False,
    filter_implant_location: str | None = None,
    filter_probe_letter: str | None = None,
    show_implant_location_query_for_all_areas: bool = False,
    show_whole_probe_tracks: bool = False,
    show_parent_brain_region: bool = False, # faster
) -> pn.pane.Matplotlib:
    
    queried_units = get_unit_location_query_df(
        filter_area=filter_area,
        filter_type=filter_type,
        case_sensitive=case_sensitive,
        include_right_hemisphere=include_right_hemisphere,
    )
    ccf_locations = (
        get_ccf_location_query_lf(
            filter_area=filter_area,
            filter_type=filter_type, 
            case_sensitive=case_sensitive,
            include_right_hemisphere=include_right_hemisphere,
            filter_implant_location=filter_implant_location,
            filter_probe_letter=filter_probe_letter,
            whole_probe=show_whole_probe_tracks,
            )
    ).collect()
    if show_implant_location_query_for_all_areas:
        if not filter_probe_letter or not filter_implant_location:
            logger.warning("Need to filter on implant location or probe letter to show insertions in other areas (else too many insertions will be displayed)")
            other_area_ccf_locations = None
        other_area_ccf_locations = (
            get_ccf_location_query_lf(
                filter_area="",
                filter_type=filter_type, 
                case_sensitive=case_sensitive,
                include_right_hemisphere=include_right_hemisphere,
                filter_implant_location=filter_implant_location,
                filter_probe_letter=filter_probe_letter,
                whole_probe=True,
            )
            .join(
                other=ccf_locations.lazy(),
                on=('session_id', 'electrode_group_name'),
                how="anti",
            )
        ).collect()
    else:
        other_area_ccf_locations = None
        
    if not filter_area: # if filter_area is empty, we just show the background brain volume
        areas = []
    if show_parent_brain_region or len(queried_units['location'].unique().to_list()) > 1:
        areas = queried_units['structure'].unique().to_list()
    else:
        areas = queried_units['location'].unique().to_list()
    logger.info(f"Adding {areas} to CCF image: {filter_area=}, {show_parent_brain_region=}")
    fig, axes = plt.subplots(1, 2)
    depth_column = {"horizontal": "ccf_dv", "coronal": "ccf_ap", "sagittal":  "ccf_ml"}
    for ax, projection in zip(axes, depth_column.keys()):
        ax.imshow(ccf_utils.get_ccf_projection(projection=projection, include_right_hemisphere=include_right_hemisphere)) # whole brain in grey
        xlims, ylims = ax.get_xlim(), ax.get_ylim()
        for area in areas:
            ax.imshow(ccf_utils.get_ccf_projection(area, projection=projection, with_opacity=True, include_right_hemisphere=include_right_hemisphere))
        # logger.info(f"Adding {ccf_locations.shape=} unit locations to {projection} image")
        # ax: plt.Axes
        # ax.imshow(
        #     ccf_utils.get_scatter_image(
        #         ccf_locations_df=ccf_locations,
        #         projection=projection, # type: ignore
        #         opacity_range=(0.8, 1.0),
        #     )
        # )
        for locations in (ccf_locations, other_area_ccf_locations):
            if locations is None:
                continue
            locations: pl.DataFrame
            scatter_df = (
                locations
                .select('ccf_ml', 'ccf_ap', 'ccf_dv')
                .select(pl.selectors.contains("ccf") & ~pl.selectors.contains(depth_column[projection]))
            )
            ax: plt.Axes
            scatter_array = scatter_df.to_numpy().T / 25
            logger.info(f"Adding {scatter_df.shape=} points on {projection} image")
            ax.scatter(
                *scatter_array,
                c='w' if locations is ccf_locations else 'k',
                s=0.05,
                alpha=1,
                edgecolors=None,
            )
            
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
    fig.tight_layout()
    return pn.pane.Matplotlib(fig, tight=True, format="svg", sizing_mode="stretch_width")

filter_type = pn.widgets.Select(name='Search type', options=['starts_with', 'contains'], value='starts_with')
random_area = np.random.choice(get_good_units_df().filter(pl.col('unit_id').is_not_null())['structure'].unique())
filter_area = pn.widgets.TextInput(name='Search brain area', value=random_area, styles={'font-weight': 'bold'})
toggle_case_sensitive = pn.widgets.Checkbox(name='Case sensitive', value=False)
select_group_by = pn.widgets.Select(name='Group by', options=['session_id', 'subject_id'], value='subject_id')
search_implant_location = pn.widgets.TextInput(name='Filter implant or hole', placeholder='e.g. "2002 E2" or "2002"')
search_probe_letter = pn.widgets.TextInput(name='Filter probe', placeholder='e.g. "A" or "B"')
toggle_right_hemisphere = pn.widgets.Checkbox(name='Include right hemisphere', value=False)
show_parent_brain_region = pn.widgets.Checkbox(name='Show parent structure in brain (faster)', value=False)
toggle_whole_probe = pn.widgets.Checkbox(name='Show complete probe tracks in brain images', value=False)
toggle_implant_location_query_for_all_areas = pn.widgets.Checkbox(name='Show insertions that hit other areas (requires filter on implant/hole)', value=False)

search_area = dict(
    filter_area=filter_area,
    filter_type=filter_type,
    case_sensitive=toggle_case_sensitive,
    include_right_hemisphere=toggle_right_hemisphere,
)
search_insertion = dict(
    filter_implant_location=search_implant_location,
    filter_probe_letter=search_probe_letter,
)
bound_unit_locations_bar = pn.bind(plot_unit_locations_bar, **search_area, group_by=select_group_by)
bound_co_recorded_structures_bar = pn.bind(plot_co_recorded_structures_bar, **search_area)
bound_all_unit_counts_bar = pn.bind(table_all_unit_counts, **search_area)
bound_holes_to_hit_areas = pn.bind(table_holes_to_hit_areas, **search_area)
bound_ccf_locations = pn.bind(
    plot_ccf_locations_2d, 
    **search_area, 
    show_whole_probe_tracks=toggle_whole_probe,
    **search_insertion,
    show_implant_location_query_for_all_areas=toggle_implant_location_query_for_all_areas,
)

plot_column_a = pn.Column(
    bound_all_unit_counts_bar,
    bound_co_recorded_structures_bar,
)
plot_column_b = pn.Column(
    bound_ccf_locations,
    bound_holes_to_hit_areas,
)
sidebar = pn.Column(
    select_group_by,
    filter_type,
    filter_area,
    toggle_case_sensitive,
    toggle_right_hemisphere,
    search_implant_location,
    search_probe_letter,
    toggle_implant_location_query_for_all_areas,
    toggle_whole_probe,
)
pn.template.MaterialTemplate(
    site="DR dashboard",
    title=__file__.split('\\')[-1].split('.py')[0].replace('_', ' ').title(),
    sidebar=[sidebar],
    main=[pn.Row(plot_column_b, plot_column_a), bound_unit_locations_bar],
).servable()

