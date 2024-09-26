import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
import random

# Q-Learning parameters and environment setup
class GridWorld:
    def __init__(self, grid_size, n_obstacles):
        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.grid = np.zeros((grid_size, grid_size))
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.obstacles = self.generate_obstacles()
        self.place_goal_and_obstacles()
        self.step_count = 0

    def generate_obstacles(self):
        obstacles = []
        while len(obstacles) < self.n_obstacles:
            obstacle = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if obstacle != self.start and obstacle != self.goal and obstacle not in obstacles:
                obstacles.append(obstacle)
        return obstacles

    def place_goal_and_obstacles(self):
        self.grid[self.goal] = 2  # Goal is marked with 2
        for obs in self.obstacles:
            self.grid[obs] = -1  # Obstacles are marked with -1

    def reset(self):
        self.step_count = 0
        return self.start

    def step(self, state, action):
        # Define the four possible actions (up, down, left, right)
        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        new_state = (state[0] + actions[action][0], state[1] + actions[action][1])

        self.step_count += 1

        # Check if the action leads to a valid state
        if new_state[0] < 0 or new_state[0] >= self.grid_size or new_state[1] < 0 or new_state[1] >= self.grid_size:
            return state, -1, False  # Out of bounds, return to the same state and penalty

        # Check if the agent hit an obstacle
        if new_state in self.obstacles:
            return new_state, -10, False  # Penalty for hitting obstacle

        # Check if the agent reached the goal
        if new_state == self.goal:
            # Calculate optimal steps to the goal using Manhattan distance
            optimal_steps = abs(self.goal[0] - self.start[0]) + abs(self.goal[1] - self.start[1])
            tolerance = 1
            bonus = 5*optimal_steps if self.step_count <= optimal_steps + tolerance else 0
            
            return new_state, 10*optimal_steps + bonus, True

        # Increase penalty for every move to minimize steps (e.g., -2 per step)
        return new_state, -5, False  # Larger penalty for each step

# Q-Learning agent
class QLearningAgent:
    def __init__(self, grid_size, learning_rate, discount_factor, exploration_rate):
        self.q_table = np.zeros((grid_size, grid_size, 4))  # Q-table
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice([0, 1, 2, 3])  # Random action
        return np.argmax(self.q_table[state[0], state[1], :])  # Exploit best action

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1], :])
        td_target = reward + self.discount_factor * self.q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.learning_rate * td_error

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout with user inputs and interval component
gridworld_layout = dbc.Container([
    html.H1("Reinforcement Learning: GridWorld Navigation with Live Movement"),

    # Top Row: Inputs and Run Simulation button
    dbc.Row([
        dbc.Col([  # Inputs (2x3 grid)
            dbc.Row([
                dbc.Col([
                    html.Label("Grid Size"),
                    dcc.Input(id='qlearn-grid-size-input', type='number', value=5, min=3, max=10, step=1),
                ], width=6),
                dbc.Col([
                    html.Label("Number of Obstacles"),
                    dcc.Input(id='qlearn-obstacles-input', type='number', value=3, min=1, max=10, step=1),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Learning Rate"),
                    dcc.Input(id='qlearn-learning-rate-input', type='number', value=0.1, min=0.01, max=1.0, step=0.01),
                ], width=6),
                dbc.Col([
                    html.Label("Discount Factor"),
                    dcc.Input(id='qlearn-discount-factor-input', type='number', value=0.9, min=0.0, max=1.0, step=0.01),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Exploration Rate"),
                    dcc.Input(id='qlearn-exploration-rate-input', type='number', value=0.2, min=0.0, max=1.0, step=0.01),
                ], width=6),
                dbc.Col([
                    html.Label("Training Episodes"),
                    dcc.Input(id='qlearn-episodes-input', type='number', value=100, min=10, max=1000, step=10),
                ], width=6),
            ])
        ], width=8),  # Left Column (8 width)

        dbc.Col([  # Run Simulation button
            html.Button('Run Simulation', id='qlearn-run-simulation-button', n_clicks=0, className='btn btn-primary', style={'margin-top': '50px'}),
        ], width=4)  # Right Column (4 width)
    ], style={'margin-bottom': '30px'}),

    # Bottom Row: Graphs (Grid and Reward/Policy)
    dbc.Row([
        # Grid Graph in the left column
        dbc.Col([
            dcc.Graph(id='qlearn-grid-graph'),
        ], width=6, style={'height': '100%'}),  # Set a fixed height to ensure it fills the left column

        # Reward and Policy graphs stacked in the right column
        dbc.Col([
            dbc.Row([
                dcc.Graph(id='qlearn-reward-graph'),
            ], style={'height': '50%'}),  # Take up half the available height
            dbc.Row([
                dcc.Graph(id='qlearn-policy-graph'),
            ], style={'height': '50%'}),  # Take up the remaining half height
        ], width=6, style={'height': '600px'})  # Match the height of the left column
    ], style={'height': '600px'}),
    dcc.Interval(id='qlearn-interval-component', interval=100, n_intervals=0, disabled=True)
], fluid=True)

# Global variables to store environment, agent, and state between intervals
env = None
agent = None
current_state = None
cumulative_rewards = []
total_reward = 0
step_count = 0
episode_count = 0


def register_gridworld_callbacks(app):
    # Combined callback to handle both starting the simulation and updating the steps
    @app.callback(
        [Output('qlearn-grid-graph', 'figure'),
        Output('qlearn-reward-graph', 'figure'),
        Output('qlearn-policy-graph', 'figure'),
        Output('qlearn-interval-component', 'disabled'),
        Output('qlearn-interval-component', 'n_intervals')],
        [Input('qlearn-run-simulation-button', 'n_clicks'),
        Input('qlearn-interval-component', 'n_intervals')],
        [State('qlearn-grid-size-input', 'value'),
        State('qlearn-obstacles-input', 'value'),
        State('qlearn-learning-rate-input', 'value'),
        State('qlearn-discount-factor-input', 'value'),
        State('qlearn-exploration-rate-input', 'value'),
        State('qlearn-episodes-input', 'value')]
    )
    def run_simulation(n_clicks, n_intervals, grid_size, n_obstacles, learning_rate, discount_factor, exploration_rate, episodes):
        global env, agent, current_state, cumulative_rewards, total_reward, step_count, episode_count
        
        # If the "Run Simulation" button is clicked, initialize the environment and agent
        if n_clicks > 0 and n_intervals == 0:
            env = GridWorld(grid_size, n_obstacles)
            agent = QLearningAgent(grid_size, learning_rate, discount_factor, exploration_rate)
            current_state = env.reset()
            cumulative_rewards = []
            total_reward = 0
            step_count = 0
            episode_count = 0
            
            # Start interval for step updates
            return update_grid_figure(), update_reward_figure(), update_policy_figure(), False, 0  # Enable interval and reset n_intervals
        
        # If the interval is active, update the simulation steps
        if env and agent and n_intervals > 0:
            action = agent.choose_action(current_state)
            next_state, reward, done = env.step(current_state, action)
            agent.learn(current_state, action, reward, next_state)
            
            current_state = next_state
            total_reward += reward
            step_count += 1
            
            # Reset environment when done or after 50 steps
            if done or step_count >= 50:
                cumulative_rewards.append(total_reward)
                total_reward = 0
                current_state = env.reset()
                episode_count += 1
                step_count = 0
            
            return update_grid_figure(), update_reward_figure(), update_policy_figure(), False, n_intervals  # Continue updating the grid
        
        return {}, {}, {}, True, n_intervals  # Keep the interval disabled until the button is clicked

    # Function to update the grid visualization
    def update_grid_figure():
        global env, current_state
        
        grid_copy = env.grid.copy()
        grid_copy[current_state] = 1  # Mark agent's position
        fig = go.Figure(go.Heatmap(z=grid_copy))
        fig.update_layout(
            title=f"GridWorld Environment - Episode {episode_count}",
            height=600,  
            autosize=False,
            margin=dict(t=30, b=30, l=30, r=30),  # Add some padding
            yaxis=dict(scaleanchor="x", scaleratio=1),  # Ensure square aspect ratio
            xaxis=dict(scaleanchor="y", scaleratio=1)   # Ensure square aspect ratio
        )
        return fig

    # Function to update the reward graph
    def update_reward_figure():
        global cumulative_rewards
        fig = go.Figure(go.Scatter(x=list(range(len(cumulative_rewards))), y=cumulative_rewards, mode='lines+markers'))
        fig.update_layout(title="Cumulative Rewards Over Episodes", xaxis_title="Episode", yaxis_title="Cumulative Reward", height=300)
        return fig

    # Function to update the policy graph (Q-values)
    def update_policy_figure():
        global agent
        q_values = np.max(agent.q_table, axis=2)  # Get the maximum Q-value for each state
        fig = go.Figure(go.Heatmap(z=q_values))
        fig.update_layout(title="Learned Policy (Q-values)", height=300)
        return fig

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=True)
