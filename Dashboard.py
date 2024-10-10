import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load your data
df = pd.read_csv('output.csv')  # Replace with the actual path to the dataset

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Lumpy Skin Disease Dashboard", style={'text-align': 'center'}),

    # Image and description
    html.Div([
        html.P("Lumpy Skin Disease (LSD) is a viral disease affecting cattle and buffalo, causing fever, nodules on the skin, and death in severe cases. This dashboard visualizes data related to the spread and environmental factors of LSD.", style={'text-align': 'center'})
    ], style={'padding': '20px'}),

    # Histogram and box plot section
    html.Div([
        html.Label("Select Feature for Histogram and Box Plot:", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': col, 'value': col} for col in df.select_dtypes(include=np.number).columns],
            value='Mean_Temp',
            style={'width': '50%', 'margin': '0 auto'}
        ),
        dcc.Graph(id='hist-box-plot', style={'margin-top': '20px'}),
    ], style={'padding': '20px', 'text-align': 'center'}),

    # Map with feature selection
    html.Div([
        html.Label("Select Feature for Map:", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='map-feature-dropdown',
            options=[{'label': col, 'value': col} for col in df.columns],
            value='Mean_Temp',
            style={'width': '50%', 'margin': '0 auto'}
        ),
        dcc.Graph(id='world-map', style={'margin-top': '20px'}),
    ], style={'padding': '20px', 'text-align': 'center'}),
    
    # Correlation heatmap section
    html.Div([
        dcc.Graph(id='correlation-heatmap'),
    ], style={'padding': '20px'}),

    # Scatter plot section
    html.Div([
        html.Label("Select Features for Scatter Plot:", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='scatter-x-dropdown',
            options=[{'label': col, 'value': col} for col in df.select_dtypes(include=np.number).columns],
            value='Mean_Temp',
            style={'width': '36%', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='scatter-y-dropdown',
            options=[{'label': col, 'value': col} for col in df.select_dtypes(include=np.number).columns],
            value='Precipitation_Amount',
            style={'width': '36%', 'display': 'inline-block', 'margin-left': '10px'}
        ),
        dcc.Graph(id='scatter-plot', style={'margin-top': '20px'}),
    ], style={'padding': '20px', 'text-align': 'center'}),

    # Feature importance section
    html.Div([
        dcc.Graph(id='feature-importance'),
    ], style={'padding': '20px'}),

    # Bi-chart for lumpy cases section
    html.Div([
        dcc.Graph(id='lumpy-barchart'),
    ], style={'padding': '20px'})
])

# Callbacks for interactive plots
@app.callback(
    [Output('hist-box-plot', 'figure'),
     Output('world-map', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('feature-importance', 'figure'),
     Output('lumpy-barchart', 'figure')],
    [Input('feature-dropdown', 'value'),
     Input('map-feature-dropdown', 'value'),
     Input('scatter-x-dropdown', 'value'),
     Input('scatter-y-dropdown', 'value')]
)
def update_graphs(selected_feature, map_feature, scatter_x, scatter_y):
    # Histogram and box plot for lumpy = 0 and lumpy = 1
    fig_hist_box = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=(f"Box Plot of {selected_feature}", 
                                                 f"Histogram of {selected_feature}"),
                                 vertical_spacing=0.1)

    df_lumpy0 = df[df['lumpy'] == 0]
    df_lumpy1 = df[df['lumpy'] == 1]

    # Box plot
    fig_hist_box.add_trace(go.Box(y=df_lumpy0[selected_feature], name='Lumpy = 0', marker_color='blue'), row=1, col=1)
    fig_hist_box.add_trace(go.Box(y=df_lumpy1[selected_feature], name='Lumpy = 1', marker_color='orange'), row=1, col=1)

    # Histogram
    fig_hist_box.add_trace(go.Histogram(x=df_lumpy0[selected_feature], name='Lumpy = 0', opacity=0.75, marker_color='blue'), row=2, col=1)
    fig_hist_box.add_trace(go.Histogram(x=df_lumpy1[selected_feature], name='Lumpy = 1', opacity=0.75, marker_color='orange'), row=2, col=1)

    fig_hist_box.update_layout(title=f"Box Plot and Histogram of {selected_feature}", height=600)

    # Map with selected feature (only where lumpy = 1)
    df_lumpy1_map = df[df['lumpy'] == 1]
    fig_map = px.scatter_mapbox(df_lumpy1_map, lat="Latitude", lon="Longitude", color=map_feature,
                                hover_data=[map_feature], zoom=2, height=500, mapbox_style="carto-positron")
    fig_map.update_layout(title=f"Map showing {map_feature} for Lumpy = 1")

    # Correlation heatmap
    corr = df.corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Blues',  # Updated color scale
        text=np.round(corr.values, 2),
        hoverinfo='text'
    ))
    fig_corr.update_layout(title="Correlation Heatmap with Values", height=600)

    # Scatter plot
    fig_scatter = px.scatter(df, x=scatter_x, y=scatter_y, color='lumpy', title=f"{scatter_x} vs {scatter_y}")

    # Feature importance using RandomForest
    X = df.drop('lumpy', axis=1)
    y = df['lumpy']
    model = RandomForestClassifier()
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_importance = px.bar(importance, title="Feature Importance for Lumpy Skin Disease", color=importance.index, color_continuous_scale=px.colors.qualitative.Set1)

    # Bi-chart for lumpy cases
    fig_barchart = px.bar(df['lumpy'].value_counts(), title="Distribution of Lumpy Cases", color=df['lumpy'].value_counts().index, color_discrete_sequence=['lightblue', 'lightcoral'])

    return fig_hist_box, fig_map, fig_corr, fig_scatter, fig_importance, fig_barchart

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
