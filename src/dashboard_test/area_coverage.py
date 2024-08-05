import logging
from typing import Iterable, Literal, TypeVar

import brainglobe_heatmap
import brainrender
import brainrender.actors
import brainrender.scene
import matplotlib.pyplot as plt
import npc_lims
import numpy as np
import panel as pn
import plotly.express as px
import polars as pl
import upath

import dashboard_test.ccf

# import iblatlas.plots
# import iblatlas.atlas


# pn.extension('vtk')   

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
                    pl.col('isi_violations_count') < 0.5,
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
        .join(
            other=dashboard_test.ccf.get_ccf_structure_tree_df().lazy(),
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
    
    units = (
        get_good_units_df()
        .lazy()
        .filter(location_expr)
        .sort('date', "location")
    ).collect()
    logger.info(f"Filtered on area, found {len(units['location'].unique())} locations across {len(units['session_id'].unique())} sessions: location.{search_type}({search_term}, {case_sensitive=})")
    return units

@pn.cache
def get_ccf_location_query_lf(
    search_term: str,
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    implant_location: str | list[str] | None = None,
    whole_probe: bool = False,
) -> pl.LazyFrame:
    
    if not implant_location:
        implant_location = None
    elif not isinstance(implant_location, str) and isinstance(implant_location, Iterable):
        _locations = list(loc for loc in implant_location if loc)
        if len(_locations) > 1:
            logger.warning(f"Multiple implant locations found: {_locations}")
        implant_location = _locations[0]
    
    queried_units = get_unit_location_query_df(
        search_term=search_term,
        search_type=search_type,
        case_sensitive=case_sensitive,
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
            pl.col('implant_location') == implant_location if implant_location else pl.lit(True),
        )
        .select('session_id', 'electrode_group_name', 'implant_location', 'ccf_ml', 'ccf_ap', 'ccf_dv', 'location', 'structure')
        .join(
            other=(
                dashboard_test.ccf.get_ccf_structure_tree_df().lazy()
                .select('acronym', pl.selectors.starts_with("color_"))
            ),
            right_on='acronym',
            left_on='location',
        )  
    )
    approx_n_unique_sessions = len(locations.select(pl.col('session_id').approx_n_unique()).collect())
    approx_n_unique_locations = len(locations.select(pl.col('location').approx_n_unique()).collect())
    logger.info(f"Filtered on area, found approx {approx_n_unique_locations} locations across {approx_n_unique_sessions} sessions: location.{search_type}({search_term}, {case_sensitive=}, {implant_location=}, {whole_probe=})")
    return locations

def plot_unit_locations_bar(
    search_term: str, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    group_by: Literal['session_id', 'subject_id'] = 'subject_id',
) -> pn.pane.Plotly:
    
    if not search_term:
        return pn.pane.Plotly(None)

    grouped_units = apply_unit_count_group_by(
        get_unit_location_query_df(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive)
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
    search_term: str, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
) -> pn.pane.Plotly:
    queried_units = (
        get_unit_location_query_df(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive)
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
        # .pipe(
        #     lambda df: df.join(
        #         other=(
        #             df
        #             .group_by('structure')
        #             .agg(
        #                 pl.col('structure').count().alias('total_structure_count')
        #                 )
        #         ),
        #         on='structure',
        #     )
        # )
        #? supposed to link to operations below, but doesn't work
    )

    other_units = (
        other_units
        .group_by(pl.col('structure', 'session_id')).agg([
            pl.col('unit_id').count().alias('unit_count'), 
        ])
        .join(
            other_units.group_by(pl.col('structure')).agg(pl.col('structure').count().alias('total_structure_count')),
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
    insertions: pl.DataFrame = (
        get_unit_location_query_df(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive)
        .group_by('session_id', 'electrode_group_name')
        .agg([
            pl.col('implant_location').first().alias("implant hole")
        ])
        .get_column('implant hole')
        .value_counts(sort=True, name="n insertions")
    )
    # Create a Panel Tabulator object
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
        theme="modern",
        page_size=5,
        pagination='local',
        # width=200,
        stylesheets=[stylesheet],
    )

@pn.cache
def get_ccf_scene() -> brainrender.Scene:
    scene = brainrender.Scene()     
    # adjust root-brain region
    scene.backend = "itkwidgets"
    scene.actors[0].opacity(0.05) 
    scene.plotter.background([1, 1, 1])
    return scene

def plot_ccf_locations_3d( 
    search_term: str, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    implant_location: str | list[str] | None = None,
    whole_probe: bool = False,
)  -> pn.pane.VTK:
    
    whole_probe = False
    
    queried_units = get_unit_location_query_df(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive)
    ccf_locations = get_ccf_location_query_lf(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive, implant_location=implant_location, whole_probe=whole_probe)
    
    scene = get_ccf_scene()     
    
    logger.info(f"Removing {len(scene.actors[1:])} actors from 3D scene")
    scene.remove(*scene.actors[1:]) # remove all actors except root-brain region
    if search_term: # if search_term is empty, regions will be every area recorded, which will be too many
        regions = queried_units['location'].drop_nulls().unique().to_list()
        logger.info(f"Adding {len(regions)} brain regions to 3D scene")
        for region in regions:
            scene.add_brain_region(
                region, 
                hemisphere='left', 
                alpha=0.1,
            )

        n = 5 if whole_probe else 1
        points = (
            ccf_locations
            .sort('session_id', 'electrode_group_name', 'ccf_dv')
            .gather_every(n)
        ).collect()
        logger.info(f"Adding {len(points)} {'electrode' if whole_probe else 'unit'} points to 3D scene (gathered every {n} rows)")
        track_points = brainrender.actors.Points(
            points.select('ccf_dv', 'ccf_ap', 'ccf_ml').to_numpy(),
            radius=15,
            res=1,
            # colors='0x000000',
            colors=points['color_hex_str'].to_list(), #! this is not working
            alpha=1,
        )
        scene.add(track_points)

    # TODO crashes when search term is updated
    # TODO unit locations are shifted/scaled
    
    # handle = plotter.show(*scene.renderables)
    # handle.objects[0].color = 0xCBD6E2 #set background brain to gray

    # # Create the graphics structure. The renderer renders into the render
    # # window.
    # ren = vtk.vtkRenderer()
    # renWin = vtk.vtkRenderWindow()
    # renWin.AddRenderer(ren)

    # # Add the actors to the renderer, set the background and size
    # for actor in scene.actors:
    #     ren.AddActor(actor)
    # ren.SetBackground(1, 1, 1)
    return pn.pane.VTK(scene.plotter.window, width=500, height=500, orientation_widget=True, interactive_orientation_widget=True) 

    # scene = brainrender.Scene()
    # scene = brainrender.scene.Scene()
    # for area in queried_units['location'].unique():
    #     scene.add_brain_region(area, hemisphere='left', alpha=0.3)
    
    # track_points = brainrender.actors.Points(
    # track_points = brainrender.scene.actors.Points(
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
   
def plot_ccf_locations_ibl( 
    search_term: str, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    implant_location: str | list[str] | None = None,
    whole_probe: bool = False,
) -> pn.pane.Matplotlib:
    queried_units = get_unit_location_query_df(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive)
    ccf_df = get_ccf_location_query_lf(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive).collect()
    
    ADJUST_X = ADJUST_Y = -5000 #! adjust for brain-globe heatmap origin at center (5000 is not correct)
    
    fig, axes = plt.subplots(1, 2)
    depth_axis = {"frontal": "ccf_ap", "horizontal": "ccf_dv", "sagittal":  "ccf_ml"}
    for ax, plane in zip(axes, depth_axis.keys()):
        iblatlas.plots.plot_scalar_on_slice(
            regions=queried_units['location'].unique().to_numpy(),
            values=np.full(len(queried_units['location'].unique()), 0.1),
            slice='top' if plane == 'frontal' else plane,
            # coord=ccf_df[depth_axis[plane]].top_k(20).median() if plane != 'frontal' else -1000,
            hemisphere="left",
            show_cbar=False,
            background='boundary',
            ba=iblatlas.atlas.AllenAtlas(),
            ax=ax,
        )
        scatter_df = (
            ccf_df
            .select('ccf_ml', 'ccf_ap', 'ccf_dv')
            .select(pl.selectors.contains("ccf") & ~pl.selectors.contains(depth_axis[plane]))
        )
        ax: plt.Axes
        if plane == "frontal":
            # brain-globe heatmap is rendering left side on right of plot so invert x-axis
            ax.invert_xaxis()
            scatter_df: pl.DataFrame = scatter_df.with_columns(
                pl.col('ccf_ml').mul(-1).alias('ccf_ml'),
            )
        ax.scatter(
            *scatter_df.to_numpy().T + [[ADJUST_X * -1 if plane == 'frontal' else 1], [ADJUST_Y]],
            c=ccf_df['color_rgb'],
            s=0.01 if whole_probe else 0.5,
            alpha=0.8,
            edgecolors=None,
        )
        logging.info(f"Plotted {len(scatter_df)} points on {plane} plane")
    fig.tight_layout()
    return pn.pane.Matplotlib(fig, tight=True)

def plot_ccf_locations_brainglobe( 
    search_term: str, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    implant_location: str | list[str] | None = None,
    whole_probe: bool = False,
) -> pn.pane.Matplotlib:
    queried_units = get_unit_location_query_df(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive)
    ccf_df = get_ccf_location_query_lf(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive).collect()
    
    ADJUST_X = ADJUST_Y = -5000 #! adjust for brain-globe heatmap origin at center (5000 is not correct)
    
    fig, axes = plt.subplots(1, 2)
    depth_axis = {"frontal": "ccf_ap", "horizontal": "ccf_dv", "sagittal":  "ccf_ml"}
    for ax, plane in zip(axes, depth_axis.keys()):
        heatmap = brainglobe_heatmap.Heatmap(
            values={area: 0 for area in queried_units['location'].unique().to_list() if area},
            orientation=plane,
            hemisphere="left",
            position=ccf_df[depth_axis[plane]].top_k(20).median() + 500, 
            vmin=0,
            vmax=1,
            format="2D",
            cmap="binary",
            thickness=500, #! this seems to have no effect
        )
        heatmap.plot_subplot(
            fig=fig,
            ax=ax,
            show_leged=False,
            hide_axes=True,
            show_cbar=False,
        )
        scatter_df = (
            ccf_df
            .select('ccf_ml', 'ccf_ap', 'ccf_dv')
            .select(pl.selectors.contains("ccf") & ~pl.selectors.contains(depth_axis[plane]))
        )
        ax: plt.Axes
        if plane == "frontal":
            # brain-globe heatmap is rendering left side on right of plot so invert x-axis
            ax.invert_xaxis()
            scatter_df: pl.DataFrame = scatter_df.with_columns(
                pl.col('ccf_ml').mul(-1).alias('ccf_ml'),
            )
        ax.scatter(
            *scatter_df.to_numpy().T + [[ADJUST_X * -1 if plane == 'frontal' else 1], [ADJUST_Y]],
            c=ccf_df['color_rgb'],
            s=0.01 if whole_probe else 0.5,
            alpha=0.8,
            edgecolors=None,
        )
        logging.info(f"Plotted {len(scatter_df)} points on {plane} plane")
    fig.tight_layout()
    return pn.pane.Matplotlib(fig, tight=True)


def plot_ccf_locations_allen( 
    search_term: str, 
    search_type: Literal['starts_with', 'contains'] = 'starts_with',
    case_sensitive: bool = True,
    implant_location: str | list[str] | None = None,
    whole_probe: bool = False,
    show_parent_brain_region: bool = False, # faster
) -> pn.pane.Matplotlib:
    
    queried_units = get_unit_location_query_df(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive)
    ccf_locations = (
        get_ccf_location_query_lf(search_term=search_term, search_type=search_type, case_sensitive=case_sensitive, implant_location=implant_location, whole_probe=whole_probe)
    ).collect()
    if not search_term: # if search_term is empty, we just show the background brain volume
        areas = []
    if show_parent_brain_region or len(queried_units['location'].unique().to_list()) > 1:
        areas = queried_units['structure'].unique().to_list()
    else:
        areas = queried_units['location'].unique().to_list()
    logger.info(f"Adding {areas} to CCF image: {search_term=}, {show_parent_brain_region=}")
    fig, axes = plt.subplots(1, 2)
    depth_column = {"horizontal": "ccf_dv", "coronal": "ccf_ap", "sagittal":  "ccf_ml"}
    for ax, projection in zip(axes, depth_column.keys()):
        ax.imshow(dashboard_test.ccf.get_ccf_projection(projection=projection))
        xlims, ylims = ax.get_xlim(), ax.get_ylim()
        for area in areas:
            ax.imshow(dashboard_test.ccf.get_ccf_projection(area, projection=projection, with_opacity=True))
        scatter_df = (
            ccf_locations
            .select('ccf_ml', 'ccf_ap', 'ccf_dv')
            .select(pl.selectors.contains("ccf") & ~pl.selectors.contains(depth_column[projection]))
        )
        ax: plt.Axes
        scatter_array = scatter_df.to_numpy().T / 25
        logger.info(f"Adding {scatter_df.shape=} points on {projection} image")
        ax.scatter(
            *scatter_array,
            c='w',
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

search_type_input = pn.widgets.Select(name='Search type', options=['starts_with', 'contains'], value='starts_with')
search_term_input = pn.widgets.TextInput(name='Search location', value='MOs', styles={'font-weight': 'bold'})
search_case_sensitive_input = pn.widgets.Checkbox(name='Case sensitive', value=False)
group_by_input = pn.widgets.Select(name='Group by', options=['session_id', 'subject_id'], value='subject_id')
select_implant_hole = pn.widgets.TextInput(name='Filter implant hole', placeholder='e.g. "2002 E2"')
show_parent_brain_region = pn.widgets.Checkbox(name='Show parent structure in brain (faster)', value=False)
whole_probe_input = pn.widgets.Checkbox(name='Show complete probe tracks in brain images', value=False)

search_input = dict(search_term=search_term_input, search_type=search_type_input, case_sensitive=search_case_sensitive_input)
bound_plot_unit_locations_bar = pn.bind(plot_unit_locations_bar, **search_input, group_by=group_by_input)
bound_plot_co_recorded_structures_bar = pn.bind(plot_co_recorded_structures_bar, **search_input)
bound_table_holes_to_hit_areas = pn.bind(table_holes_to_hit_areas, **search_input)
bound_plot_ccf_locations = pn.bind(
    plot_ccf_locations_allen, 
    **search_input, 
    whole_probe=whole_probe_input,
    implant_location=select_implant_hole,
    show_parent_brain_region=show_parent_brain_region,    
)

#TODO bind ccf_locations with bound_table_holes_to_hit_areas().selected_dataframe['implant_location'].values

# column of plots   
column = pn.Column(
    bound_plot_unit_locations_bar,  
    bound_table_holes_to_hit_areas,
)
sidebar = pn.Column(
    group_by_input,
    search_type_input,
    search_term_input,
    search_case_sensitive_input,
    select_implant_hole,
    # show_parent_brain_region,
    whole_probe_input,
)
pn.template.MaterialTemplate(
    site="Dynamic Routing dashboard",
    title=__file__.split('\\')[-1].split('.py')[0].replace('_', ' ').title(),
    sidebar=[sidebar],
    main=[pn.Row(column, pn.Column(bound_plot_ccf_locations, bound_plot_co_recorded_structures_bar))],
).servable()

