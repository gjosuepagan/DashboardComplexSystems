import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np

# Function to compute the logistic map
def logistic_map(r, x):
    return r * x * (1 - x)


# Function to calculate logistic map iterations for a given r
def calculate_logistic_map(r, num_iter=500, last=100):
    x = np.random.random()  # Start with a random value of x
    results = []

    # Iterate the logistic map num_iter times
    for i in range(num_iter):
        x = logistic_map(r, x)
        
        # After reaching steady state, collect the last few iterations
        if i >= (num_iter - last):
            results.append(x)
    
    return results

# Create the Dash app
app = dash.Dash(__name__)

# App layout
logistic_layout = html.Div([
    html.H1("Logistic Map Bifurcation: Dynamic Visualization"),

    # Slider for r value
    dcc.Slider(
        id='r-slider',
        min=2.5,
        max=4.0,
        step=0.01,
        value=2.5,  # Default starting value
        marks={i: f'{i:.1f}' for i in np.arange(2.5, 4.1, 0.5)},
        tooltip={"placement": "bottom", "always_visible": True},
    ),
    
    # Graph output
    dcc.Graph(id='logistic-map-graph'),

    # Store to keep track of points for all r values less than or equal to the current r
    dcc.Store(id='data-store', data={}),

    # Label showing current r value
    html.Div(id='r-value-label', style={'marginTop': 20})
])

def register_logisticmap_callbacks(app):
    # Callback to update the graph based on the slider's r value
    @app.callback(
        Output('logistic-map-graph', 'figure'),
        Output('r-value-label', 'children'),
        Output('data-store', 'data'),
        Input('r-slider', 'value'),
        State('data-store', 'data')
    )
    def update_graph(r, data_store):
        # If there is no data stored yet, initialize an empty dictionary
        if data_store is None:
            data_store = {}

        # Calculate logistic map points for the current r value if not already stored
        if str(r) not in data_store:
            points = calculate_logistic_map(r)
            data_store[str(r)] = points

        # Create the figure
        fig = go.Figure()

        # Plot all previously stored points for r values <= current r
        for stored_r, points in data_store.items():
            if float(stored_r) <= r:
                fig.add_trace(go.Scatter(
                    x=[float(stored_r)] * len(points),
                    y=points,
                    mode='markers',
                    marker=dict(size=3, color='blue'),
                    name=f'r = {stored_r}'
                ))

        # Customize the layout
        fig.update_layout(
            title=f'Logistic Map Behavior for r ≤ {r:.3f}',
            xaxis_title='r (Complexity Parameter)',
            yaxis_title='x (Stable Points)',
            xaxis_range=[2.5, 4.0],
            yaxis_range=[0, 1],
            showlegend=False,
        )

        # Update label to show current r value
        label = f"Current r: {r:.3f}"

        return fig, label, data_store

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)
