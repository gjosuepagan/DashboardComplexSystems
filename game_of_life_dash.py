import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Initialize Dash app
app = dash.Dash(__name__)

# Default grid size; to be adjusted
DEFAULT_GRID_SIZE = 50

# Define complex initial conditions including random and a mix of complex patterns
initial_conditions_options = {
    
    "Gosper Glider Gun": [  # Indefinite gliders
        [(5, 1), (5, 2), (6, 1), (6, 2), (5, 11), (6, 11), (7, 11), 
         (4, 12), (8, 12), (3, 13), (3, 14), (9, 13), (9, 14), (6, 15), 
         (4, 16), (8, 16), (5, 17), (6, 17), (7, 17), (6, 18), (3, 21), 
         (4, 21), (5, 21), (3, 22), (4, 22), (5, 22), (2, 23), (6, 23), 
         (1, 25), (2, 25), (6, 25), (7, 25), (3, 35), (4, 35), (3, 36), (4, 36)]
    ],
    "Pulsar": [  # A larger oscillator
        [(12, 14), (13, 14), (14, 14), (18, 14), (19, 14), (20, 14), (10, 16), (15, 16), (17, 16), (22, 16), 
         (10, 17), (15, 17), (17, 17), (22, 17), (10, 18), (15, 18), (17, 18), (22, 18), (12, 20), (13, 20), 
         (14, 20), (18, 20), (19, 20), (20, 20), (15, 22), (17, 22), (12, 23), (13, 23), (14, 23), (18, 23), 
         (19, 23), (20, 23)]
    ],
    "Random": None,  # Random will be generated dynamically based on grid size
    "Complex Mix": [  # Combos of multiple patterns 
        [(5, 1), (5, 2), (6, 1), (6, 2), (5, 11), (6, 11), (7, 11), 
         (4, 12), (8, 12), (3, 13), (3, 14), (9, 13), (9, 14), (6, 15), 
         (4, 16), (8, 16), (5, 17), (6, 17), (7, 17), (6, 18), (3, 21), 
         (4, 21), (5, 21), (3, 22), (4, 22), (5, 22), (2, 23), (6, 23), 
         (1, 25), (2, 25), (6, 25), (7, 25), (3, 35), (4, 35), (3, 36), (4, 36)],  # Glider Gun
        [(12, 14), (13, 14), (14, 14), (18, 14), (19, 14), (20, 14), 
         (10, 16), (15, 16), (17, 16), (22, 16), (10, 17), (15, 17), 
         (17, 17), (22, 17), (10, 18), (15, 18), (17, 18), (22, 18), 
         (12, 20), (13, 20), (14, 20), (18, 20), (19, 20), (20, 20), 
         (15, 22), (17, 22), (12, 23), (13, 23), (14, 23), (18, 23), 
         (19, 23), (20, 23)],  # Pulsar
    ]
}

# Initialize the grid with custom initial conditions
def initialize_grid(size, initial_conditions):
    grid = np.zeros((size, size), dtype=int)
    if initial_conditions is None:
        # Use random initial state directly
        return np.random.choice([0, 1], size * size, p=[0.8, 0.2]).reshape(size, size)
    else:
        # Initialize grid based on pattern coordinates
        for pattern in initial_conditions:
            for (x, y) in pattern:
                grid[x % size, y % size] = 1
        return grid

# Update the grid based on Conway's rules
def update_grid(grid):
    new_grid = grid.copy()
    grid_size = grid.shape[0]
    for row in range(grid_size):
        for col in range(grid_size):
            total = int((grid[row, (col-1)%grid_size] + grid[row, (col+1)%grid_size] +
                         grid[(row-1)%grid_size, col] + grid[(row+1)%grid_size, col] +
                         grid[(row-1)%grid_size, (col-1)%grid_size] + grid[(row-1)%grid_size, (col+1)%grid_size] +
                         grid[(row+1)%grid_size, (col-1)%grid_size] + grid[(row+1)%grid_size, (col+1)%grid_size]))
            
            if grid[row, col] == 1:
                if total < 2 or total > 3:
                    new_grid[row, col] = 0
            else:
                if total == 3:
                    new_grid[row, col] = 1
    return new_grid

# Generate an image of the current grid state
def generate_image(grid):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap='binary')
    ax.set_title("Conway's Game of Life")
    ax.axis('off')
    
    # Convert plot to PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

# Layout of the Dash app
game_of_life_layout = html.Div([
    html.H1("Game of Life Dashboard"),
    dcc.Input(
        id='gol-grid-size-input',
        type='number',
        value=DEFAULT_GRID_SIZE,
        min=10,
        max=100,
        step=10,
        placeholder="Grid Size",
        style={'margin-bottom': '20px'}
    ),
    dcc.Dropdown(
        id='gol-initial-condition-dropdown',
        options=[{'label': key, 'value': key} for key in initial_conditions_options.keys()],
        value='Gosper Glider Gun'
    ),
    html.Button("Start Simulation", id='gol-start-button', n_clicks=0),
    dcc.Interval(
        id='gol-interval-component',
        interval=200,  # Updating interval in milliseconds
        n_intervals=0,
        disabled=True  
    ),
    html.Img(id='game-of-life-image'),
    dcc.Store(id='gol-grid-store'),  
    dcc.Store(id='gol-current-grid-size')  
])

def register_gol_callbacks(app):
    # Unified callback to handle starting, updating, and resetting the simulation
    @app.callback(
        [Output('game-of-life-image', 'src'),
         Output('gol-grid-store', 'data'),
         Output('gol-interval-component', 'disabled'),
         Output('gol-current-grid-size', 'data'),
         Output('gol-start-button', 'n_clicks')],
        [Input('gol-start-button', 'n_clicks'),
         Input('gol-interval-component', 'n_intervals'),
         Input('gol-initial-condition-dropdown', 'value')],
        [State('gol-grid-store', 'data'),
         State('gol-grid-size-input', 'value'),
         State('gol-current-grid-size', 'data')]
    )
    def update_simulation(n_clicks, n_intervals, selected_condition, grid_data, grid_size, current_grid_size):
        ctx = dash.callback_context
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]

        # Check if the dropdown was changed, reset simulation in that case
        if triggered == 'gol-initial-condition-dropdown':
            initial_conditions = initial_conditions_options[selected_condition]
            grid = initialize_grid(grid_size, initial_conditions)
            image = generate_image(grid)
            return image, grid.tolist(), True, grid_size, 0  # Reset n_clicks to 0 and disable interval

        # Initialize the grid when the button is clicked
        elif triggered == 'gol-start-button' and n_clicks > 0:
            initial_conditions = initial_conditions_options[selected_condition]
            grid = initialize_grid(grid_size, initial_conditions)
            image = generate_image(grid)
            return image, grid.tolist(), False, grid_size, n_clicks  # Start the simulation

        # Update the grid on each interval tick
        elif triggered == 'gol-interval-component' and grid_data is not None:
            grid = np.array(grid_data)
            if current_grid_size and current_grid_size != grid.shape[0]:
                grid = initialize_grid(current_grid_size, [])  
            updated_grid = update_grid(grid)
            image = generate_image(updated_grid)
            return image, updated_grid.tolist(), False, current_grid_size, n_clicks  # Continue updating

        # Default case before any action
        return dash.no_update, dash.no_update, True, dash.no_update, n_clicks  



# # Running the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=True)
