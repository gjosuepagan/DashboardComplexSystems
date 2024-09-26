import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Import GridWorld and Network dashboards
from ReinforcementLearning_dash import gridworld_layout, register_gridworld_callbacks
from NetworkAnalysis_dash import network_layout, register_network_callbacks
from ABM_SchellingSakoda_dash import abm_layout, register_abm_callbacks
from logistic_map_dash2 import logistic_layout, register_logisticmap_callbacks
from game_of_life_dash import game_of_life_layout, register_gol_callbacks

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Main layout with tabs for each dashboard
app.layout = dbc.Container([
    html.H1("Modeling Complex Systems in Data Science"),

    # Tabs for navigating between different dashboards
    dcc.Tabs(id="tabs", value='gameoflife-tab', children=[
        dcc.Tab(label='Game of Life', value='gameoflife-tab'),
        dcc.Tab(label='Logistical Map', value='logistic-tab'),
        dcc.Tab(label='Agent-Based Modeling', value='abm-tab'),
        dcc.Tab(label='Network Analysis', value='network-tab'),
        dcc.Tab(label='Q-Learning Simulation', value='gridworld-tab'),
    ]),
    
    # Placeholder to load the content of the selected tab
    html.Div(id='tabs-content')
])

# Callback to switch between dashboards
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'gridworld-tab':
        return gridworld_layout
    elif tab == 'network-tab':
        return network_layout
    elif tab == 'abm-tab':
        return abm_layout
    elif tab == 'logistic-tab':
        return logistic_layout
    elif tab == 'gameoflife-tab':
        return game_of_life_layout

# Register callbacks for each dashboard
register_gol_callbacks(app)
register_gridworld_callbacks(app)
register_network_callbacks(app)
register_abm_callbacks(app)
register_logisticmap_callbacks(app)


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
