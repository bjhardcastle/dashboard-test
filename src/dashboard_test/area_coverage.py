import html
import logging
from typing import Literal, TypeVar

import brainrender
import brainrender.actors
import npc_lims
import panel as pn
import plotly.express as px
import polars as pl
import vedo

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@pn.cache
def get_component_lazyframe(nwb_component: str) -> pl.LazyFrame:
    path = npc_lims.get_cache_path(
        nwb_component, 
        version='v0.0.231', 
        consolidated=True,
    )
    logger.info(f"Reading dataframe from {path}")
    return pl.scan_parquet(path)

@pn.cache
def get_good_units_lazyframe() -> pl.LazyFrame:

    location_lf = (
        get_component_lazyframe("session")
        .filter(
            pl.col('keywords').list.contains('templeton').not_()
        )
        .join(
            other=(
                get_component_lazyframe("performance")
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
                get_component_lazyframe("units")
                .filter(
                    pl.col('isi_violations_count') < 0.5,
                    pl.col('amplitude_cutoff') < 0.1,
                    pl.col('presence_ratio') > 0.95,
                )
            ),
            on='session_id',
        )
        .join(
            other=(
                get_component_lazyframe('electrode_groups')
                    .rename({
                        'name': 'electrode_group_name',
                        'location': "implant_location",
                    })
                    .select('session_id', 'electrode_group_name', 'implant_location')
            ),
            on=('session_id', 'electrode_group_name'),
            how="full",
        )
        .group_by([
            pl.col('session_id'),
            pl.col('location'),
        ])
        .agg([
            pl.col('location').count().alias('location_count'),
            pl.col('structure').first(),
            pl.col('subject_id').first(),
            pl.col('date').first(),
            pl.col('unit_id').explode().alias('unit_ids'),
            pl.col('implant_location').first(),
            pl.col('electrode_group_name').first(),
            pl.col('ccf_ap').first(),
            pl.col('ccf_dv').first(),
            pl.col('ccf_ml').first(),
            pl.col('peak_channel').first(),
        ])
    )    
    logger.info("Fetched all unit locations")
    return location_lf


DataFrameOrLazyFrame = TypeVar("DataFrameOrLazyFrame", pl.DataFrame, pl.LazyFrame)
def apply_location_query(
    df_or_lf: DataFrameOrLazyFrame,
    search_term: str, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
) -> DataFrameOrLazyFrame:
    
    if not case_sensitive:
        search_term = search_term.lower()
        location_col = pl.col('location').str.to_lowercase()
    else:
        location_col = pl.col('location')
    location_expr = getattr(location_col.str, search_type)(search_term)
    
    df_or_lf = df_or_lf.filter(location_expr)
    return df_or_lf

@pn.cache
def get_location_query_df(
    search_term: str, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
) -> pl.DataFrame:
    df = (
        apply_location_query(
            get_good_units_lazyframe(),
            search_term=search_term,
            search_type=search_type,
            case_sensitive=case_sensitive,
        )
        .sort('date', "location")
    ).collect()
    logger.info(f"Found {len(df)} units: location.{search_type}({search_term}, {case_sensitive=})")
    return df

def get_ccf_locations_lazyframe(
    search_term: str,
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    whole_probe: bool = False,
):

    electrodes_lf = (
        apply_location_query(
            get_component_lazyframe('electrodes'),
            search_term=search_term,
            search_type=search_type,
            case_sensitive=case_sensitive,
        )
        .rename({
            'x': 'ccf_ap',
            'y': 'ccf_dv',
            'z': 'ccf_ml',
            'group_name': 'electrode_group_name',
        })
        .join(
            other=(
                get_component_lazyframe('electrode_groups')
                    .rename({
                        'name': 'electrode_group_name',
                        'location': "implant_location",
                    })
                    .select('session_id', 'electrode_group_name', 'implant_location')
            ),
            on=('session_id', 'electrode_group_name'),
        )
    )
    if whole_probe: 
        return electrodes_lf
    return (
        electrodes_lf
        .join(
            other=get_good_units_lazyframe(),
            left_on=('session_id', 'electrode_group_name', 'channel'),
            right_on=('session_id', 'electrode_group_name', 'peak_channel'),    
            how="semi", # only keep rows in left table (electrodes) that have match in right table (ie selected units)
        ) 
    )

@pn.cache
def get_ccf_structure_tree() -> pl.DataFrame:
    return pl.read_csv('//allen/programs/mindscope/workgroups/np-behavior/ccf_structure_tree_2017.csv')

def plot_unit_locations_bar(
    search_term: str, 
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
        logger.warning("No structures found")
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
    search_term: str, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
) -> pn.pane.Plotly:
    query_df = get_location_query_df(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive)
    lf = (
        get_good_units_lazyframe()
        .lazy()
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
        lf
        .group_by(pl.col('structure', 'session_id')).agg([
            pl.col('unit_id').count().alias('unit_count'), 
        ])
        .join(
            other=lf.group_by(pl.col('structure')).agg(pl.col('structure').count().alias('total_structure_count')),
            on='structure',
        )
        .group_by('structure').agg(
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
        df,
        x="structure", 
        y="unit_count",
        hover_data="session_id",
        labels={'unit_count': 'units'}, 
        title=f"top {k} structures co-recorded in sessions with {search_term}", 
        barmode='group',
    ) 
    fig.update_layout(
        autosize=True,
        width=None,
    )
    return pn.pane.Plotly(fig)

def table_holes_to_hit_areas(
    search_term: str, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
) -> pn.pane.Plotly:
    df = get_location_query_df(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive)
    return pn.widgets.Tabulator(
        df['implant_location'].value_counts(sort=True).to_pandas(),
        disabled=True,
        selectable=False,
        show_index=False,
        page_size=10,
        pagination='local',
        theme="modern",
    )

def plot_ccf_locations(
    search_term: str, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    implant_location: str | None = None,
    whole_probe: bool = False,
):
    
    query_df = get_location_query_df(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive)

    lf = get_ccf_locations_lazyframe(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive, whole_probe=whole_probe)
    if implant_location:
        lf = lf.filter(pl.col('implant_location') == implant_location)
    lf = lf.filter(
        pl.col('ccf_ap').is_not_null() > -1,
        pl.col('ccf_dv').is_not_null() > -1,
        pl.col('ccf_ml').is_not_null() > -1,
    )
    ccf_df  = lf.collect()
    import brainglobe_heatmap
    
    depth_axis = {"frontal": "ccf_ap", "horizontal": "ccf_dv", "sagittal": "ccf_ml"}
    plane = "frontal"
    frontal, horizontal = [
        brainglobe_heatmap.Heatmap(
            values={area: 0 for area in query_df['location'].unique()},
            orientation=plane,
            hemisphere="left",
            position=ccf_df[depth_axis[plane]].mean(), 
            vmin=0,
            vmax=1,
            format="2D",
        )
        for plane in ["frontal", "horizontal"]
    ]
    
    # scene = brainrender.Scene()
    # for area in areas_to_show:
    #     scene.add_brain_region(area, hemisphere='left', alpha=0.3)
    
    # track_points = brainrender.actors.Points(
    #     coords.to_numpy() * 25, 
    #     radius=30,
    #     res=1,
    # )
    # scene.add(track_points)

    # count = len(coords)

    # vedo.embedWindow('k3d')
    # scene.jupyter = True

    # plt = vedo.Plotter()
    # handle = plt.show(*scene.renderables)
    # handle.objects[0].color = 0xCBD6E2 #set background brain to gray

    # #set brain region colors
    # for ia, a in enumerate(areas_to_show):

    #     color = int(get_ccf_structure_tree().filter(acronym=a)['color_hex_triplet'][0], 16)
    #     handle.objects[ia + 1].color = color
    #     handle.objects[ia + 1].opacity = 0.1

    # #set probe colors
    # track_colors = [0x030303]*count
    
    # for c, color in zip(range(count), track_colors):
    #     handle.objects[1 + len(areas_to_show) + c].color = color
    #     handle.objects[1 + len(areas_to_show) + c].opacity = 0.1
        
    # html_content = handle.get_snapshot()
    # escaped_html = html.escape(html_content)

    # # Create iframe embedding the escaped HTML and display it
    # iframe_html = f'<iframe srcdoc="{escaped_html}" style="height:100%; width:100%" frameborder="0"></iframe>'

    # # Display iframe in a Panel HTML pane
    # return pn.pane.HTML(iframe_html, height=350, sizing_mode="stretch_width")
   


# add a dropdown selector for the search type and a text input for the search term
search_type_input = pn.widgets.Select(name='Search type', options=['starts_with', 'contains'], value='starts_with')
search_term_input = pn.widgets.TextInput(name='Search location', value='MOs')
search_case_sensitive_input = pn.widgets.Checkbox(name='Case sensitive', value=False)
group_by_input = pn.widgets.Select(name='Group by', options=['session_id', 'subject_id'], value='subject_id')


search_input = dict(search_term=search_term_input, search_type=search_type_input, case_sensitive=search_case_sensitive_input)
bound_plot_unit_locations_bar = pn.bind(plot_unit_locations_bar, **search_input, group_by=group_by_input)
bound_plot_co_recorded_structures_bar = pn.bind(plot_co_recorded_structures_bar, **search_input)
bound_table_holes_to_hit_areas = pn.bind(table_holes_to_hit_areas, **search_input)
bound_plot_ccf_locations = pn.bind(plot_ccf_locations, **search_input)
# bottom row of less-important plots
bottom_row = pn.Row(bound_plot_co_recorded_structures_bar, bound_plot_ccf_locations, bound_table_holes_to_hit_areas)

# column of plots
column = pn.Column(bound_plot_unit_locations_bar, bottom_row)


pn.template.MaterialTemplate(
    site="Dynamic Routing",
    title="units by location",
    sidebar=[group_by_input, search_type_input, search_term_input, search_case_sensitive_input],
    main=[column],
).servable()

