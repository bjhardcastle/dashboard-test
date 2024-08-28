import concurrent.futures
import contextlib
import functools
import json
import logging
import pathlib

import aind_session
import npc_session
import pandas as pd
import polars as pl
import panel as pn

pn.config.theme = "dark"
pn.config.notifications = True
pn.extension("tabulator")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

EPHYS = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot")
assert EPHYS.exists()
UPLOAD = pathlib.Path("//allen/programs/mindscope/workgroups/np-exp/codeocean")
assert UPLOAD.exists()
executor = concurrent.futures.ThreadPoolExecutor()

@pn.cache
def get_folder_df():
    folders = [p.name for p in EPHYS.glob('DRpilot*') if p.is_dir() and '366122' not in p.name]
    columns = (
        "subject",
        "ephys",
        "date",
        "folder",
        "started",
        "aind ID",
        "finished",
    )
    records = []
    logger.info(f"Submitting {len(folders)} jobs to threadpool")
    def get_row(s: str):
        logger.info(f"Fetching info for {s}")
        row = dict.fromkeys(columns, None)
        row["folder"] = s
        row["ephys"] = (EPHYS / s / s).exists()
        row["date"] = npc_session.extract_isoformat_date(s)
        row["subject"] = str(npc_session.extract_subject(s))
        upload = UPLOAD / s / "upload.csv"
        row["started"] = upload.exists()
        if row["started"]:
            df = pl.read_csv(upload)
            if any('acq-datetime' in c for c in df.columns):
                for c in df.columns:
                    if 'acq-datetime' in c:
                        dt = df[c].drop_nulls()[0]
                        break
            elif r'acq-datetime\r' in df.columns:
                dt = df[r'acq-datetime\r'].drop_nulls()[0]
            elif 'acq-date' in df.columns:
                dt = f"{df['acq-date'].drop_nulls()[0]}_{df['acq-time'].drop_nulls()[0]}"
            else:
                raise ValueError(f"no datetime column found in {upload}") 
            if not dt:
                import pdb; pdb.set_trace()
            dt = dt.replace(':', '-').replace(' ', '_')
            row["aind ID"] = npc_session.AINDSessionRecord(f"ecephys_{row['subject']}_{dt}").id
            row["finished"] = aind_session.Session(row["aind ID"]).is_uploaded
        return row
    for row in executor.map(get_row, folders):
        records.append(row)
    if not records:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(records).sort_values("date", ascending=False)
    return df
    # def content_fn(row) -> pn.pane.Str:
    #     try:
    #         output = (
    #             aind_session.Session(row["aind ID"]).ecephys.sorted_data_dir / "output"
    #         ).read_text()

    #     except AttributeError:
    #         txt = "no sorted data found"
    #     else:
    #         txt = (
    #             f"raw asset ID: {aind_session.Session(row['session']).raw_data_asset.id}\n"
    #             f"sorted asset ID: {aind_session.Session(row['session']).ecephys.sorted_data_asset.id}\n"
    #             f"\noutput:\n{output}"
    #         )
    #     return pn.pane.Str(
    #         object=txt,
    #         styles={"font-size": "12pt"},
    #         sizing_mode="stretch_width",
    #     )
    
def get_folder_table(
    ephys_only: bool = False,
    unstarted_only: bool = False,
) -> pn.widgets.Tabulator:
    """Get sessions for a specific subject and date range."""
    yield pn.indicators.LoadingSpinner(
        value=True, size=20, name="Fetching folders..."
    )
    df = get_folder_df()
    if ephys_only:
        df = df[df["ephys"]]
    if unstarted_only:
        df = df[~df["started"]]
    column_filters = {
        'subject': {'type': 'input', 'func': 'like', 'placeholder': 'like x'},
        'folder': {'type': 'input', 'func': 'like', 'placeholder': 'like x'},
        'started': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
        'finished': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
    }
    # Custom formatter to highlight cells with False in the success column
    def color_negative_red(val):
        """
        Takes a scalar and returns a string with
        the css property `'color: red'` for negative
        bools, black otherwise.
        """
        color = (
            "red" if not val else ("white" if pn.config.theme == "dark" else "black")
        )
        return "color: %s" % color

    stylesheet = """
    .tabulator-cell {
        font-size: 12px;
    }
    """
    table = pn.widgets.Tabulator(
        hidden_columns=["date"] + (["ephys"] if ephys_only else []),
        # groupby=["subject"],
        value=df,
        selectable='checkbox-single',
        # disabled=True,
        show_index=False,
        sizing_mode="stretch_width",
        # row_content=content_fn,
        widths={'folder': "20%", 'aind ID': "20%"},

        embed_content=False,
        stylesheets=[stylesheet],
        formatters={
            "bool": {"type": "tickCross"},  # not working
        },
        header_filters=column_filters,
    )
    table.style.map(color_negative_red)

    # def callback(event):
    #     if event.column == "trigger":
    #         try_run_sorting(df["session"].iloc[event.row])
    #     ## expand row
    #     # else:
    #     #     table.expanded = [event.row] if event.row not in table.expanded else []
    #     #     table._update_children()

    # table.on_click(callback)
    yield table


def try_run_sorting(session_id: str) -> None:
    """Run sorting for a specific session, if it hasn't already been run."""
    session = aind_session.Session(session_id)
    try:
        _ = session.raw_data_asset
    except (FileNotFoundError, AttributeError):
        logger.info(
            f"Failed to find raw data asset for {session_id}: cannot run trigger capsule"
        )
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
        logger.info(
            f"Skipping sorting for {session_id} because it has already been triggered"
        )
        pn.state.notifications.warning("Sorting already triggered")
        return
    try:
        session.ecephys.run_sorting()
    except Exception as e:
        logger.error(f"Failed to launch trigger capsule for {session_id}: {e}")
        pn.state.notifications.error(f"Failed to run trigger capsule:\n{e!r}")
    else:
        logger.info(
            f"Launched trigger capsule for {session_id} with raw data asset {session.raw_data_asset.id}"
        )
        pn.state.notifications.success("Launched trigger capsule")


def app():
    width = 150
    usage_info = pn.pane.Alert(
        """
        """,
        alert_type="info",
        sizing_mode="stretch_width",
    )
    sidebar = pn.Column(
    )

    # bound_get_session_df = pn.bind(
    #     get_sessions_table, subject_id, specific_date, start_date, end_date
    # )
    toggle_ephys = pn.widgets.Checkbox(name="show ephys only", value=True)
    upload_button = pn.widgets.Button(name="upload selected", button_type="primary")
    return pn.template.MaterialTemplate(
        site="DR dashboard",
        title=pathlib.Path(__file__).stem.replace("_", " ").lower(),
        sidebar=[toggle_ephys, upload_button],
        main=[pn.bind(get_folder_table, toggle_ephys), usage_info],
        sidebar_width=width + 30,
    )


app().servable()
