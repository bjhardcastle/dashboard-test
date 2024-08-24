import contextlib
import logging

import panel as pn
import pandas as pd
import aind_session

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

latest_sorted_asset_ids: list[str] = []
"""For tracking which sessions have been triggered for sorting this
runtime."""

def get_sessions(
    subject_id: str,
    specific_date: str,
    start_date: str,
    end_date: str,
) -> tuple[aind_session.Session, ...]:
    """Get sessions for a specific subject and date range."""
    params = {k:v for k,v in locals().items() if v}
    logger.info(f"Fetching ecephys sessions with {params}")
    if specific_date:
        date_args = {
            "date": specific_date,
        }
    else:
        date_args = {
            "start_date": start_date,
            "end_date": end_date,
        }
    sessions = []
    for s in subject_id.replace(' ', '').split(','):
        with contextlib.suppress(LookupError):
            sessions.extend(
                aind_session.get_sessions(subject_id=s, platform='ecephys', **date_args)
            )
    logger.info(f"Found {len(sessions)} sessions")
    return tuple(sessions)

def get_sessions_table(
    subject_id: str,
    specific_date: str,
    start_date: str,
    end_date: str,
) -> pn.widgets.Tabulator:
    """Get sessions for a specific subject and date range."""
    if not (subject_id or specific_date or start_date or end_date):
        sessions = ()
    else:
        sessions = get_sessions(subject_id, specific_date, start_date, end_date)
    columns = []
    for s in sessions:
        logger.info(f"Fetching info for {s.id}")
        row = dict.fromkeys(
            ('session', 'raw asset', 'latest sorted asset', 'success', 'probes'),
            None
        )
        row["session"] = s.id
        with contextlib.suppress(Exception):
            row["raw asset"] = s.raw_data_asset.id
        with contextlib.suppress(Exception):
            row["latest sorted asset"] = s.ecephys.sorted_data_asset.id
        with contextlib.suppress(Exception):
            row["success"] = not s.ecephys.is_sorting_fail
        with contextlib.suppress(Exception):
            row["probes"] = s.ecephys.sorted_probes
        columns.append(row)
    df = pd.DataFrame(columns)
    
    def content_fn(row) -> pn.pane.Str:
        try:
            txt = (aind_session.Session(row['session']).ecephys.sorted_data_dir / 'output').read_text()
        except:
            txt = 'no sorted data found'
        return pn.pane.Str(
            object=txt,
            styles={'font-size': '12pt'},
            sizing_mode='stretch_width',
        )

    stylesheet = """
    .tabulator-cell {
        font-size: 12px;
    }
    """
    table = pn.widgets.Tabulator(
        value=df,
        selectable=False,
        show_index=False,
        layout='fit_columns',
        sizing_mode='stretch_width',
        row_content=content_fn,
        embed_content=True,
        stylesheets=[stylesheet],
        buttons={
            'trigger': '<i class="fa fa-sync" title="re-run sorting"></i>',
        }
    )
    table.on_click(
        
        lambda e: try_run_sorting(df['session'].iloc[e.row])
    )
    return table

def try_run_sorting(session_id: str) -> None:
    """Run sorting for a specific session, if it hasn't already been run."""
    session = aind_session.Session(session_id)
    try:
        _ = session.raw_data_asset
    except (FileNotFoundError, AttributeError):
        logger.info(f"Failed to find raw data asset for {session_id}: cannot run trigger capsule")
        return
    try:
        latest_sorted_asset = session.ecephys.sorted_data_asset
    except AttributeError:
        latest_sorted_asset = None
    # if sorting has been run, the previous data asset 
    if latest_sorted_asset and latest_sorted_asset.id in latest_sorted_asset_ids:
        logger.info(f"Skipping {session_id} because it has already been triggered this runtime")
        return
    try:
        session.ecephys.run_sorting()
    except Exception as e:
        logger.error(f"Failed to launch trigger capsule for {session_id}: {e}")
    else:
        logger.info(f"Launched trigger capsule for {session_id} with raw data asset {session.raw_data_asset.id}")
        if latest_sorted_asset is not None: latest_sorted_asset_ids.append(latest_sorted_asset.id)


width = 150
subject_id = pn.widgets.TextInput(name="Subject ID(s)", value="728537", placeholder="comma separated", width=width)
specific_date = pn.widgets.TextInput(name="Specific date", value="", width=width)
start_date = pn.widgets.TextInput(name="Start date", value="", width=width)
end_date = pn.widgets.TextInput(name="End date", value="", width=width)

sidebar = pn.Column(
    subject_id,
    specific_date,
    start_date,
    end_date,
)

bound_get_session_df = pn.bind(get_sessions_table, subject_id, specific_date, start_date, end_date)

pn.template.MaterialTemplate(
    site="ecephys dashboard",
    title=__file__.split('\\')[-1].split('.py')[0].replace('_', ' '),
    sidebar=[sidebar],
    main=[bound_get_session_df],
    sidebar_width=width + 30,
).servable()
