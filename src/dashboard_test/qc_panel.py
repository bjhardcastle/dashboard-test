from typing import Generator

import boto3
import bs4
import pandas as pd
import panel as pn
import upath
from bokeh.models.widgets.tables import NumberFormatter, BooleanFormatter


PROJECT_PATH = upath.UPath("s3://aind-scratch-data/dynamic-routing")
LOCAL_PATH = upath.UPath("//allen/programs/mindscope/workgroups/dynamicrouting")
QC_ROOT_PATH = PROJECT_PATH / "qc"
QC_ROOT_PATH = LOCAL_PATH / "qc"

DOC_HTML = """
<!DOCTYPE html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="description" content="">
        <title></title>
        <h1 text-align="middle"></h1>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="">
    </head>
    <body></body>
</html>
"""

FIG_TAG = """
<figure>
  <img src="" alt=""></img>
  <figcaption><b><a href="" target="_blank"></a></b></figcaption>
</figure>
"""

JSON_TAG = """
<figure>
  <iframe src="" title=""></iframe>
  <figcaption><b><a href="" target="_blank"></a></b></figcaption>
</figure>
"""

TEXT_TAG = """
<figure>
    <p></p>
</figure>
"""

def get_session_table() -> pd.DataFrame:
    return pd.read_parquet(PROJECT_PATH / "session_metadata" / "sessions.parquet").sort_values("session_id", ascending=False)

def get_session_qc_paths(session_id: str) -> Generator[upath.UPath, None, None]:
    for path in QC_ROOT_PATH.rglob(session_id + "*"):
        if not path.is_dir():
            yield path

def get_session_qc_urls(session_id: str) -> Generator[str, None, None]:
    """
    >>> assert next(get_session_qc_urls("636766_2023-01-23"))
    """
    for path in get_session_qc_paths(session_id):
        yield get_presigned_url(path)
        
def get_presigned_url(path: upath.UPath) -> str:
    bucket = path.parts[0]
    key = path.as_posix().split(bucket)[-1]
    url = boto3.client("s3").generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket.strip("/"), "Key": key},
        ExpiresIn=24 * 3600,
    )
    return url

def get_session_qc_html(session_id: str) -> str:
    doc = bs4.BeautifulSoup(DOC_HTML, features='html.parser')
    # doc.head.title.append(f'{session} {session_qc_root.parent.name}')
    # doc.head.h1.append(f'{session} {session_qc_root.parent.name}')
    # doc.head.link['href'] = f'{CSS_FILENAME}.css'

    def fmt(string: str) -> str:
        return string.replace('_', ' ').strip()

    def add_section(p: upath.Path, heading_idx: int = 1) -> bs4.Tag:
        if p.parent == session_qc_root:
            parent = doc.head
        else:
            parent = doc.body
        parent.append(title := doc.new_tag(f'h{heading_idx}'))
        title.append(fmt(p.name) if p.is_dir() else fmt(p.parent.name))
        parent.append(section := doc.new_tag('div', attrs={'class': 'row'}))
        return section

    def add_figure(p: pathlib.Path, section: bs4.Tag) -> None:
        section.append(div := doc.new_tag('div', attrs={'class': 'column'}))
        fig = bs4.BeautifulSoup(FIG_TAG, features='html.parser')
        fig.img['src'] = fig.img['alt'] = p
        fig.figcaption.b.a.append(fmt(p.stem))
        # fig.figcaption.b.a['href'] = f'file:///{p}'
        div.append(fig)

    def add_json(p: pathlib.Path, section: bs4.Tag) -> None:
        if 'plotly' in p.name:
            return
        div = doc.new_tag('div', attrs={'class': 'column'})
        section.append(div)
        json = bs4.BeautifulSoup(JSON_TAG, features='html.parser')
        json.iframe['src'] = json.iframe['title'] = p
        json.figcaption.b.a.append(fmt(p.stem))
        # json.figcaption.b.a['href'] = f'file:///{p}'
        div.append(json)

    def add_text(p: pathlib.Path, section: bs4.Tag) -> None:
        div = doc.new_tag('div', attrs={'class': 'column'})
        section.append(div)
        text = bs4.BeautifulSoup(TEXT_TAG, features='html.parser')
        fig.figcaption.b.a.append(fmt(p.stem))
        text.p.append(p.read_text())
        div.append(text)
        
    def add_qc_contents(path: pathlib.Path, idx: int = 1):
        """Recursively add subfolders and their files to html doc"""
        section = None
        for p in path.iterdir():
            if '.vscode' in p.parts:
                continue
            if p.is_dir():
                add_qc_contents(p, idx + 1)
            else:
                if not section:
                    section = add_section(p, idx)
                if p.suffix == '.png':
                    add_figure(p, section)
                elif p.suffix == '.json':
                    add_json(p, section)

    add_qc_contents(session_qc_root)
    return str(doc)
    
@pn.cache
def row_content_fn(row: pd.Series):
    session_id = row['session_id'][:-2]
    print(session_id)
    path = next(get_session_qc_paths(session_id))
    print(path)
    url = get_presigned_url(path)
    fig = bs4.BeautifulSoup(FIG_TAG, features='html.parser')
    fig.img['src'] = fig.img['alt'] = url
    # fig.figcaption.b.a.append(path.stem)
    # fig.figcaption.b.a['href'] = Rf'file://{path}'
    return pn.pane.HTML(str(fig))


session_table_widget = pn.widgets.Tabulator(
    get_session_table(),
    layout="fit_data_stretch",
    # sizing_mode="fit_columns",
    row_content=row_content_fn,
    embed_content=False,
    formatters={
        'float': NumberFormatter(format='0.00'),
        'bool': BooleanFormatter(),
    },
    disabled=True,
    selectable=False,
)

pn.template.FastListTemplate(
    site="Panel",
    title="Dynamic Routing QC",
    main=[session_table_widget],
    accent='#00A170',
    main_layout=None,
    main_max_width="100%",
).servable()


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )