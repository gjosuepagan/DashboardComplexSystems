import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# Agent class for the disease spread model
class DiseaseAgent(Agent):
    def __init__(self, unique_id, model, health_status):
        super().__init__(unique_id, model)
        self.health_status = health_status  # "healthy" or "diseased"

    def step(self):
        # Move the agent randomly, considering the temperature
        self.move(self.model.temperature)
        # Update the health status of the agent based on neighbors
        self.update_health()

    def move(self, temperature):
        # Determine the movement range based on temperature
        if temperature > 1.0:
            # For higher temperatures, allow agents to move farther away
            distance = int(temperature)  # Temperature determines the range of movement
            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=distance)
        else:
            # At lower temperatures, restrict movement to nearby cells
            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)

        new_position = self.random.choice(possible_moves)
        self.model.grid.move_agent(self, new_position)

    def update_health(self):
        diseased_neighbors = 0
        total_neighbors = 0

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        for neighbor in neighbors:
            total_neighbors += 1
            if neighbor.health_status == "diseased":
                diseased_neighbors += 1

        # Determine if the agent becomes diseased based on the ratio of diseased neighbors
        if total_neighbors > 0:
            disease_spread_ratio = diseased_neighbors / total_neighbors
            if self.health_status == "healthy" and disease_spread_ratio >= self.model.disease_threshold:
                self.health_status = "diseased"

# Disease spread model class
class DiseaseModel(Model):
    def __init__(self, N, width, height, disease_threshold, initial_diseased_percentage=0.5, temperature=1.0):
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.disease_threshold = disease_threshold
        self.temperature = temperature  # Temperature controls agent movement

        # Calculate how many agents start off as diseased
        num_diseased = int(self.num_agents * initial_diseased_percentage)
        num_healthy = self.num_agents - num_diseased

        # Create healthy agents
        for i in range(num_healthy):
            agent = DiseaseAgent(i, self, "healthy")
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Create diseased agents
        for i in range(num_healthy, self.num_agents):
            agent = DiseaseAgent(i, self, "diseased")
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Data collector to track the percentage of diseased agents
        self.datacollector = DataCollector(
            {"diseased": lambda model: self.count_diseased_agents() / self.num_agents}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def count_diseased_agents(self):
        diseased_agents = [agent for agent in self.schedule.agents if agent.health_status == "diseased"]
        return len(diseased_agents)

# Dash app setup
app = dash.Dash(__name__)

# App layout
abm_layout = html.Div([
    html.H1("Disease Spread Model"),
    
    # Input fields with titles
    html.Div([
        html.Div([
            html.Label("Number of Agents"),
            dcc.Input(id='abm-num-agents-input', type='number', value=200, min=10, max=400, step=1)
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label("Disease Spread Threshold (0-1)"),
            dcc.Input(id='abm-disease-threshold-input', type='number', value=1, min=0.0, max=1.0, step=0.05)
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label("Initial Diseased Percentage (0-1)"),
            dcc.Input(id='abm-initial-diseased-input', type='number', value=0.5, min=0.0, max=1.0, step=0.05)
        ], style={'margin-bottom': '10px'}),
        html.Div([
            html.Label("Movement Range (0.1-10)"),
            dcc.Input(id='abm-temperature-input', type='number', value=10, min=0.1, max=10.0, step=0.1)
        ], style={'margin-bottom': '10px'}),
    ], style={'margin-bottom': '20px'}),

    # Run and Stop buttons
    html.Button('Run Simulation', id='abm-run-simulation-button', n_clicks=0, style={'margin-right': '10px'}),
    html.Button('Stop Simulation', id='abm-stop-simulation-button', n_clicks=0),

    # Flexbox container to display the graphs side by side
    html.Div([
        dcc.Graph(id='abm-grid-graph', style={'width': '49%', 'display': 'inline-block'}),
        dcc.Graph(id='abm-disease-graph', style={'width': '49%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between'}),

    # Interval for the simulation
    dcc.Interval(id='abm-interval-component', interval=500, n_intervals=0, disabled=True)
])

# Initialize model
model = None
max_steps = 100

def register_abm_callbacks(app):
    # Combined callback to handle both starting and stopping the simulation
    @app.callback(
        [Output('abm-interval-component', 'disabled'),
        Output('abm-interval-component', 'n_intervals')],
        [Input('abm-run-simulation-button', 'n_clicks'),
        Input('abm-stop-simulation-button', 'n_clicks')],
        [State('abm-num-agents-input', 'value'),
        State('abm-disease-threshold-input', 'value'),
        State('abm-initial-diseased-input', 'value'),
        State('abm-temperature-input', 'value')]
    )
    def control_simulation(run_clicks, stop_clicks, num_agents, disease_threshold, initial_diseased_percentage, temperature):
        global model
        ctx = dash.callback_context

        # Determine which button triggered the callback
        if not ctx.triggered:
            return True, 0  # Ensure the interval is disabled by default
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == 'abm-run-simulation-button':
                # Initialize the model and start the simulation
                model = DiseaseModel(num_agents, 20, 20, disease_threshold, initial_diseased_percentage, temperature)
                return False, 0  # Enable the interval component and reset step count to 0
            elif button_id == 'abm-stop-simulation-button':
                return True, 0  # Disable the interval component and reset step count to 0

        return True, 0

    # Callback to update the model and plots at each interval tick
    @app.callback(
        [Output('abm-grid-graph', 'figure'),
        Output('abm-disease-graph', 'figure')],
        [Input('abm-interval-component', 'n_intervals')]
    )
    def update_model(n_intervals):
        global model
        if model is None or n_intervals >= max_steps:
            return {}, {}  # Stop updating after max_steps

        # Step the model
        model.step()

        # Prepare agent positions for plotting
        agent_x = []
        agent_y = []
        agent_color = []

        for agent in model.schedule.agents:
            x, y = agent.pos
            agent_x.append(x)
            agent_y.append(y)
            if agent.health_status == "healthy":
                agent_color.append("green")  # Green for healthy
            else:
                agent_color.append("red")  # Red for diseased

        # Create scatter plot for the agents with gridlines
        grid_figure = go.Figure(go.Scatter(
            x=agent_x, y=agent_y,
            mode='markers',
            marker=dict(size=10, color=agent_color),
            text=["Healthy" if color == "green" else "Diseased" for color in agent_color]
        ))
        grid_figure.update_layout(
            title=f'Agent Grid at Step {n_intervals}',
            xaxis=dict(range=[0, 20], showgrid=True, dtick=1),  # Gridlines every value from 0 to 20
            yaxis=dict(range=[0, 20], showgrid=True, dtick=1),  # Gridlines every value from 0 to 20
            margin=dict(t=30, b=0, l=0, r=0)
        )

        # Diseased agents graph
        history = model.datacollector.get_model_vars_dataframe()
        disease_figure = go.Figure(go.Scatter(x=history.index, y=history['diseased'], mode='lines'))
        disease_figure.update_layout(title='Percentage of Diseased Agents Over Time',
                                    xaxis_title='Steps', yaxis_title='Diseased Agents (%)')

        return grid_figure, disease_figure

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=True)
