import dash
from dash import html, dcc, Input, Output
import pandas as pd
import pickle
import plotly.express as px

# Charger le modèle et les données
with open("real_estate_model.pkl", "rb") as f:
    model = pickle.load(f)

features_columns = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']

data_df = pd.read_csv("Real_Estate.csv")
data_df['Predicted Price'] = model.predict(data_df[features_columns])

#  Initialiser Dash 
app = dash.Dash(__name__)
app.title = "Dashboard Immobilier"

app.layout = html.Div(style={'font-family':'Arial', 'padding':'20px'}, children=[

    html.H1("🏠 Dashboard Prédiction Immobilière", style={'text-align':'center', 'color':'#007BFF'}),

    html.Div(style={'display':'flex', 'gap':'20px'}, children=[

        # Partie gauche: Inputs
        html.Div(style={'flex':'1', 'padding':'10px'}, children=[
            html.Div(style={'background-color':'#e6f7ff', 'padding':'20px', 'border-radius':'10px'}, children=[
                html.H3("Paramètres d'entrée"),
                
                html.Label("Distance à la station MRT (m)"),
                dcc.Slider(id='distance_slider', min=0, max=1000, step=10, value=500,
                           marks={0:'0',500:'500',1000:'1000'}),
                
                html.Label("Nombre de magasins"),
                dcc.Slider(id='stores_slider', min=0, max=20, step=1, value=5,
                           marks={0:'0',10:'10',20:'20'}),
                
                html.Label("Latitude"),
                dcc.Input(id='latitude_input', type='number', value=24.98, step=0.01, style={'width':'100%'}),
                
                html.Label("Longitude"),
                dcc.Input(id='longitude_input', type='number', value=121.54, step=0.01, style={'width':'100%'}),
            ])
        ]),

        # Partie droite: Graphiques et prédiction
        html.Div(style={'flex':'2', 'padding':'10px'}, children=[
            html.Div(id='prediction_output', style={'text-align':'center', 'font-size':'24px', 'margin-bottom':'20px'}),

            dcc.Graph(id='price_hist'),
            dcc.Graph(id='price_3d'),

            html.H3("Tableau des données avec prédictions"),
            dash.dash_table.DataTable(
                id='table_output',
                columns=[{"name": i, "id": i} for i in data_df.columns],
                data=data_df.to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor':'#007BFF','color':'white'},
                style_cell={'textAlign':'center'}
            )
        ])
    ])
])

#  Callback pour mettre à jour tout en temps réel 
@app.callback(
    Output('prediction_output', 'children'),
    Output('price_hist', 'figure'),
    Output('price_3d', 'figure'),
    Input('distance_slider', 'value'),
    Input('stores_slider', 'value'),
    Input('latitude_input', 'value'),
    Input('longitude_input', 'value')
)
def update_dashboard(distance, stores, lat, long):
    # Prédiction
    features_input = pd.DataFrame([[distance, stores, lat, long]], columns=features_columns)
    prediction = model.predict(features_input)[0]

    # Histogramme des prix
    hist_fig = px.histogram(data_df, x='Predicted Price', nbins=20,
                            title="Distribution des Prix Prévus",
                            color_discrete_sequence=['#007BFF'])
    hist_fig.update_layout(plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9')

    # Graphique 3D
    fig3d = px.scatter_3d(data_df, x='Latitude', y='Longitude', z='Predicted Price',
                          color='Predicted Price', color_continuous_scale='Viridis',
                          title="Prix Prévus vs Latitude / Longitude")
    fig3d.update_layout(scene=dict(xaxis_title='Latitude', yaxis_title='Longitude', zaxis_title='Prix'))

    return f"Prix prédit pour vos paramètres : {prediction:.2f}", hist_fig, fig3d

if __name__=="__main__":
    app.run(debug=True)
