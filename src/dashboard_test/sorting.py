import contextlib
import logging
import pathlib

import npc_session
import panel as pn
import pandas as pd
import aind_session

pn.config.theme = 'dark'
pn.config.notifications = True
pn.extension('tabulator')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    subjects = {npc_session.extract_subject(s) for s in subject_id.split(',')}
    sessions = set()
    for subject in subjects:
        if subject is None:
            continue
        s = aind_session.get_sessions(
                subject_id=subject,
                platform='ecephys',
                **date_args,
            )
        if not s:
            logger.info(f"No sessions found for {subject}")
            pn.state.notifications.warning(f"No sessions found for {subject}")
        else:
            sessions.update(s)
    logger.info(f"Found {len(sessions)} sessions")
    return tuple(sessions)

def get_sessions_table(
    subject_id: str,
    specific_date: str,
    start_date: str,
    end_date: str,
) -> pn.widgets.Tabulator:
    """Get sessions for a specific subject and date range."""
    yield pn.indicators.LoadingSpinner(value=True, size=20, name='Fetching data from CodeOcean...')
    if not (subject_id or specific_date or start_date or end_date):
        sessions = ()
    else:
        sessions = get_sessions(subject_id, specific_date, start_date, end_date)
    columns = ('subject', 'session', 'raw asset', 'latest sorted asset', 'jobs running', 'success', 'probes')
    records = []
    current_sorting_computations = aind_session.search_computations(
        capsule_or_pipeline_id=aind_session.ecephys.SORTING_PIPELINE_ID,
        in_progress=True,
        ttl_hash=aind_session.get_ttl_hash(20),
    )
    # import pdb; pdb.set_trace()
    for s in sessions:
        logger.info(f"Fetching info for {s.id}")
        row = dict.fromkeys(
            columns,
            None
        )
        row["session"] = s.id
        row["subject"] = s.subject_id
        with contextlib.suppress(AttributeError):
            row["raw asset"] = s.raw_data_asset.id
            row["latest sorted asset"] = s.ecephys.sorted_data_asset.id
            row["jobs running"] = [c.name for c in current_sorting_computations if any(asset.id == s.raw_data_asset.id for asset in c.data_assets)]
        with contextlib.suppress(Exception):
            row["success"] = int(not s.ecephys.is_sorting_fail)
            row["probes"] = s.ecephys.sorted_probes
        records.append(row)
    if not records:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(records).sort_values('session')
    
    def content_fn(row) -> pn.pane.Str:
        try:
            output = (
                aind_session.Session(row['session']).ecephys.sorted_data_dir / 'output'
            ).read_text()

        except AttributeError:
            txt = 'no sorted data found'
        else:
            txt = (
            	f"raw asset ID: {aind_session.Session(row['session']).raw_data_asset.id}\n"
                f"sorted asset ID: {aind_session.Session(row['session']).ecephys.sorted_data_asset.id}\n"
                f"\noutput:\n{output}"
            )
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
        hidden_columns=['subject', 'raw asset', 'latest sorted asset'],
        groupby=['subject'],
        value=df,
        selectable=False,
        #disabled=True,
        show_index=False,
        sizing_mode='stretch_width',
        row_content=content_fn,
        embed_content=False,
        stylesheets=[stylesheet],
        formatters= {
            'bool': {'type': 'tickCross'} # not working
        },     
        buttons={
            'trigger': '<i class="fa fa-sync" title="re-run sorting"></i>',
        }
    )
    def callback(event):
        if event.column == 'trigger':
            try_run_sorting(df['session'].iloc[event.row])
        else:
            # table.row_content(df.iloc[event.row])
            table.expanded = [event.row] if event.row not in table.expanded else []
            table._update_children()
    table.on_click(callback)
    yield table

def try_run_sorting(session_id: str) -> None:
    """Run sorting for a specific session, if it hasn't already been run."""
    session = aind_session.Session(session_id)
    capsule_url = "https://codeocean.allenneuraldynamics.org/capsule/6726080/tree"
    try:
        _ = session.raw_data_asset
    except (FileNotFoundError, AttributeError):
        logger.info(f"Failed to find raw data asset for {session_id}: cannot run trigger capsule")
        pn.state.notifications.error("No raw data asset found")
        return
    currently_running_computations = aind_session.search_computations(
        capsule_or_pipeline_id=aind_session.ecephys.TRIGGER_CAPSULE_ID,
        in_progress=True,
        ttl_hash=aind_session.utils.get_ttl_hash(10),
    )
    currently_sorting_asset_id_to_name = {
        parameter.value: computation.name
        for computation in currently_running_computations
        for parameter in computation.parameters
        if parameter.name == "input data asset ID"
        or parameter.param_name == "input_data_asset_ID"
    }
    if session.raw_data_asset.id in currently_sorting_asset_id_to_name:
        logger.info(f"Skipping sorting for {session_id} because it has already been triggered")
        pn.state.notifications.warning(f"Sorting already triggered")
        return
    try:
        computation = session.ecephys.run_sorting()
    except Exception as e:
        logger.error(f"Failed to launch trigger capsule for {session_id}: {e}")
        pn.state.notifications.error(f"Failed to run trigger capsule:\n{e!r}")
    else:
        logger.info(f"Launched trigger capsule for {session_id} with raw data asset {session.raw_data_asset.id}")
        pn.state.notifications.success(f"Launched trigger capsule")

def app():
    width = 150
    subject_id = pn.widgets.TextInput(name="Subject ID(s)", value="", placeholder="comma separated", width=width)
    specific_date = pn.widgets.TextInput(name="Specific date", value="", width=width)
    start_date = pn.widgets.TextInput(name="Start date", value="", width=width)
    end_date = pn.widgets.TextInput(name="End date", value="", width=width)
    usage_info = pn.pane.Alert(
        """
        ## Usage
        - Enter a subject ID, or multiple IDs separated by commas
        - Expand a row in the table to view the contents of the "output" file from the session's latest sorted data asset
        - `fail` indicates whether the latest sorted data asset shows signs of failure (based on number of files, certain words in the output file, etc.)
        - `probes` indicates which probes successfully completed sorting in the latest sorted data asset
        - Press the "reload" icon in the right-most column to run sorting for the session (via the trigger capsule)
            - If the trigger capsule is currently running for the session it won't be re-run
            - The sorting pipeline itself is not checked
        - Most data from CodeOcean is cached with a time-to-live of 10 minutes
        
        Trigger capsule: https://codeocean.allenneuraldynamics.org/capsule/6726080/tree
        Sorting pipeline: https://codeocean.allenneuraldynamics.org/capsule/8510735/tree
        """,
        alert_type='info',
        sizing_mode="stretch_width",
    )
    sidebar = pn.Column(
        subject_id,
        specific_date,
        start_date,
        end_date,
    )

    bound_get_session_df = pn.bind(get_sessions_table, subject_id, specific_date, start_date, end_date)

    return pn.template.MaterialTemplate(
        site="DR dashboard",
        title=pathlib.Path(__file__).stem.replace('_', ' ').lower(),
        sidebar=[sidebar],
        main=[bound_get_session_df, usage_info],
        sidebar_width=width + 30,
    )
    
app().servable()
