import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import networkx as nx
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc

# Function to simulate influence spread in a network
def simulate_influence_spread(G, initial_influence, influence_prob, steps):
    # Initialize influenced status for nodes
    influenced = {node: False for node in G.nodes()}
    initial_nodes = np.random.choice(G.nodes(), size=int(initial_influence * len(G)), replace=False)
    
    for node in initial_nodes:
        influenced[node] = True

    # Record the number of influenced nodes over time
    influence_counts = [sum(influenced.values())]

    for _ in range(steps):
        new_influenced = influenced.copy()
        for node in G.nodes():
            if not influenced[node]:  # If not yet influenced
                neighbors = list(G.neighbors(node))
                if any(influenced[neighbor] for neighbor in neighbors):  # Check if any neighbor is influenced
                    if np.random.rand() < influence_prob:
                        new_influenced[node] = True
        influenced = new_influenced
        influence_counts.append(sum(influenced.values()))

    return influenced, influence_counts

# Function to generate a random network
def generate_network(n_nodes, connection_prob):
    G = nx.erdos_renyi_graph(n_nodes, connection_prob)
    return G

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
network_layout = dbc.Container([
    html.H1("Network Analysis: Influence Spread in a Social Network"),

    # Input section: single row with two columns (inputs on left, button on right)
    dbc.Row([
        dbc.Col([  # First column for the inputs
            html.Label("Number of Nodes"),
            dcc.Input(id='network-num-nodes-input', type='number', value=100, min=10, max=500, step=10, style={'margin-bottom': '10px'}),

            html.Label("Initial Influenced Percentage"),
            dcc.Input(id='network-initial-influence-input', type='number', value=0.1, min=0.0, max=1.0, step=0.05, style={'margin-bottom': '10px'}),

            html.Label("Number of Simulation Steps"),
            dcc.Input(id='network-num-steps-input', type='number', value=10, min=1, max=100, style={'margin-bottom': '10px'}),

            html.Label("Connection Probability"),
            dcc.Input(id='network-connection-prob-input', type='number', value=0.05, min=0.01, max=1.0, step=0.01, style={'margin-bottom': '10px'}),

            html.Label("Influence Spread Probability"),
            dcc.Input(id='network-influence-prob-input', type='number', value=0.2, min=0.0, max=1.0, step=0.05, style={'margin-bottom': '10px'}),
        ], width=8),  # Adjust width to have more space for inputs

        dbc.Col([  # Second column for the Run Simulation button
            html.Button('Run Simulation', id='network-run-simulation-button', n_clicks=0, className='btn btn-primary', style={'margin-top': '40px'}),
        ], width=4),
    ], style={'margin-bottom': '20px'}),

    # Plot section: split into two columns
    dbc.Row([
        # First column for Network Influence Spread
        dbc.Col([
            dcc.Graph(id='network-graph'),
        ], width=6),

        # Second column with two rows for Influenced Nodes and Degree Distribution
        dbc.Col([
            dbc.Row([
                dcc.Graph(id='network-influence-graph'),
            ], style={'height': '50%'}),
            dbc.Row([
                dcc.Graph(id='network-degree-dist-graph'),
            ], style={'height': '50%'}),
        ], width=6)
    ]),
], fluid=True)

def register_network_callbacks(app):
    # Callback to generate and simulate the network
    @app.callback(
        [Output('network-graph', 'figure'),
        Output('network-influence-graph', 'figure'),
        Output('network-degree-dist-graph', 'figure')],
        [Input('network-run-simulation-button', 'n_clicks')],
        [State('network-num-nodes-input', 'value'),
        State('network-connection-prob-input', 'value'),
        State('network-initial-influence-input', 'value'),
        State('network-influence-prob-input', 'value'),
        State('network-num-steps-input', 'value')]
    )
    def update_graph(n_clicks, num_nodes, connection_prob, initial_influence, influence_prob, num_steps):
        if n_clicks == 0:
            return {}, {}, {}  # No output before simulation
        
        # Generate network
        G = generate_network(num_nodes, connection_prob)
        
        # Simulate influence spread
        influenced, influence_counts = simulate_influence_spread(G, initial_influence, influence_prob, num_steps)
        
        # Network visualization
        pos = nx.spring_layout(G)  # Layout for network visualization
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='gray'), hoverinfo='none', mode='lines')

        node_x = []
        node_y = []
        node_color = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_color.append(1 if influenced[node] else 0)  # 1 for influenced, 0 for not
            node_text.append(f'Node {node}: {"Influenced" if influenced[node] else "Not Influenced"}')

        node_trace = go.Scatter(
            x=node_x, y=node_y, text=node_text, mode='markers', hoverinfo='text',
            marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=node_color, colorbar=dict(thickness=15))
        )

        network_fig = go.Figure(data=[edge_trace, node_trace])
        network_fig.update_layout(showlegend=False, hovermode='closest', title="Network Influence Spread",
                                margin=dict(t=30, b=0, l=0, r=0), height=600)

        # Plot of influenced nodes over time
        influence_fig = go.Figure(go.Scatter(x=list(range(num_steps + 1)), y=influence_counts, mode='lines+markers'))
        influence_fig.update_layout(title="Influenced Nodes Over Time", xaxis_title="Steps", yaxis_title="Influenced Nodes", height=300)

        # Degree distribution
        degrees = [degree for _, degree in G.degree()]
        degree_fig = go.Figure(go.Histogram(x=degrees, nbinsx=20))
        degree_fig.update_layout(title="Node Degree Distribution", xaxis_title="Degree", yaxis_title="Count", height=300)

        return network_fig, influence_fig, degree_fig

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=True)
