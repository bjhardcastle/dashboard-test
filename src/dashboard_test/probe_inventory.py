import datetime
import pandas as pd
import sqlite3
import panel as pn
import pathlib

# Database connection setup
DATABASE_PATH = 'probes.db'  # Replace with your actual database path

# Function to initialize the database
def initialize_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS probes (
            serial_number TEXT PRIMARY KEY,
            date_installed TEXT DEFAULT NULL,
            date_removed TEXT DEFAULT NULL,
            issue TEXT DEFAULT NULL,
            rig TEXT DEFAULT NULL,
            storage_location TEXT DEFAULT NULL,
            is_destroyed BOOLEAN DEFAULT FALSE
        )
    ''')
    conn.commit()
    conn.close()

# Function to load data from the database
def load_data():
    conn = sqlite3.connect(DATABASE_PATH)
    query = "SELECT * FROM probes"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Function to save data to the database
def save_data(df):
    conn = sqlite3.connect(DATABASE_PATH)
    df.to_sql("probes", conn, if_exists="replace", index=False)
    conn.close()

# Initialize the database
initialize_database()

# Load initial data
df = load_data()

# Create an editable DataFrame widget
editable_df = pn.widgets.Tabulator(df)

# Callback function to save changes to the database
def save_changes(event):
    updated_df = editable_df.value
    save_data(updated_df)

# # Button to save changes
# save_button = pn.widgets.Button(name="Save Changes", button_type="primary")
# save_button.on_click(save_changes)

# Form fields for new entry
serial_number_input = pn.widgets.TextInput(name='Serial Number')
date_installed_input = pn.widgets.DatePicker(name='Date Installed', value=datetime.datetime.now())
date_removed_input = pn.widgets.DatePicker(name='Date Removed')
issue_input = pn.widgets.TextInput(name='Issue')
rig_input = pn.widgets.TextInput(name='Rig')
storage_location_input = pn.widgets.TextInput(name='Storage Location')
is_destroyed_input = pn.widgets.Checkbox(name='Is Destroyed')

# Callback function to add new entry
def add_entry(event):
    new_entry = {
        'serial_number': serial_number_input.value,
        'rig': rig_input.value,
        'date_installed': date_installed_input.value.strftime('%Y-%m-%d'),
        'date_removed': date_removed_input.value.strftime('%Y-%m-%d') if date_removed_input.value else None,
        'issue': issue_input.value or None,
        'storage_location': storage_location_input.value or None,
        'is_destroyed': is_destroyed_input.value,
    }
    global df
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    save_data(df)
    editable_df.value = df  # Update the editable DataFrame widget

# Button to add new entry
add_button = pn.widgets.Button(name="Add Entry", button_type="success")
add_button.on_click(add_entry)

editable_df.on_edit(save_changes)

# Form layout
form = pn.Column(
    serial_number_input,
    rig_input,
    date_installed_input,
    date_removed_input,
    issue_input,
    storage_location_input,
    is_destroyed_input,
    add_button
)

# Layout the dashboard
# dashboard = pn.Column("# Probe Tracking Dashboard", editable_df)
pn.template.MaterialTemplate(
    site="DR dashboard",
    title=pathlib.Path(__file__).stem.replace('_', ' ').lower(),
    sidebar=[form],
    main=[editable_df],
    # sidebar_width=200,
).servable()