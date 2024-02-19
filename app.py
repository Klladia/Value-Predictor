from dash import Dash, dcc, html, callback_context
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc

import plotly.express as px
import pandas as pd
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
from itertools import zip_longest

import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import webbrowser




app = Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])




makes_list = ['BMW', 'Audi', 'Volkswagen', 'Ford', 'Opel', 'Mercedes-Benz', 'Toyota', 'Renault',
              'Skoda', 'Peugeot', 'Kia', 'Hyundai', 'Volvo', 'Citroën', 'Nissan', 'Mazda', 'Seat',
              'Fiat', 'Honda', 'Suzuki', 'Dacia', 'Jeep', 'Mitsubishi', 'Porsche', 'MINI', 'Land Rover',
              'Lexus', 'Alfa Romeo', 'Chevrolet', 'Dodge', 'Jaguar', 'Cupra', 'Subaru', 'Chrysler',
              'SsangYong', 'Tesla', 'DS Automobiles', 'Infiniti', 'Maserati', 'Saab', 'Smart', 'MG',
              'RAM', 'Lancia', 'Cadillac', 'Isuzu', 'Daihatsu', 'Bentley', 'Aixam', 'Ferrari', 'Ligier',
              'Abarth', 'Aston Martin', 'Lamborghini', 'Lincoln', 'BMW-ALPINA', 'GMC', 'Daewoo', 'Microcar',
              'Hummer', 'Lada', 'Rolls-Royce', 'Maxus', 'Baic', 'Buick', 'McLaren', 'Rover', 'Iveco',
              'Polonez', 'Pontiac', 'Acura', 'MAN', 'Maybach', 'BYD', 'Polestar', 'Alpine', 'Genesis']


input_data = pd.DataFrame()
raw_df = pd.DataFrame()



app.layout = html.Div([
    
    dbc.Card([
        dbc.Row([
            html.Br(),
            html.H1("CAR VALUE PREDICTOR", style = {'font-size': '50px'}),
            html.H2("which is all you need to determine the value of your car", style = {'font-size': '30px', 'color': 'black'}),
        ]),
        #html.Hr(), 
        dbc.Row([
            dbc.Col(dbc.Card(html.Div([
                dbc.Row([
                    dbc.Col(
                        html.Div(id='checkbox1-output'),
                    width=1),
                    dbc.Col(
                        dcc.Markdown("#### &nbsp;What's the make of your car?"),
                    width=6),
                    dbc.Col(
                        dcc.Dropdown(id='makes-dropdown', 
                                     options=makes_list,  
                                     searchable=True,
                                     placeholder="It's a ..........",
                                     style={'border': 'none', 'outline': 'none'}),
                    width=4),
                    dbc.Col(width=1)
                ]),
                html.Br(),
                
                dbc.Row([
                    dbc.Col(
                        html.Div(id='checkbox2-output'),
                    width=1),  
                    dbc.Col(
                        dcc.Markdown("#### &nbsp;And the model is...?"),
                    width=5),
                    dbc.Col(
                        dcc.Input(id='model-input',
                                  placeholder="Enter the model",
                                  style={'border': 'none', 'text-align': 'center', 'outline': 'none', 'background-color': 'rgb(220, 220, 220)'}),
                    width=3),
                    dbc.Col(width=3)
                ]),

                html.Br(),
                
                dbc.Row([
                    dbc.Col(
                        html.Div(id='checkbox-year-output'),
                    width=1),  
                    dbc.Col(
                        dcc.Markdown("#### &nbsp;The car was manufactured in "),
                    width=6),
                    dbc.Col(
                        dcc.Input(id='year-input',
                                  placeholder="2015?",
                                  style={'border': 'none', 'text-align': 'center', 'outline': 'none', 'background-color': 'rgb(220, 220, 220)'}),
                    width=3),
                    dbc.Col(width=2)
                ]),

                html.Br(),
                
                
                dbc.Row([
                    dbc.Col(
                        html.Div(id='checkbox3-output'),
                    width=1),
                    dbc.Col(
                        dcc.Markdown("#### &nbsp;So far, your car has already driven"),
                    width=7),
                    dbc.Col(
                        dcc.Input(id='mileage-input',
                                  placeholder=". . . . . . . . . ",
                                  style={'border': 'none', 'text-align': 'center', 'outline': 'none', 'background-color': 'rgb(220, 220, 220)'}),
                    width=3),
                    dbc.Col(
                        dcc.Markdown(" km"),
                    width=1),
                
                ]),
                html.Br(),
                
                dbc.Row([
                    dbc.Col(
                        html.Div(id='checkbox4-output'),
                    width=1),
                    dbc.Col(
                        dcc.Markdown("#### &nbsp;Engine capacity is "), 
                    width=4),
                    dbc.Col(
                        dcc.Input(id='capacity-input',
                                  placeholder="4.0 ?",
                                  style={'border': 'none', 'text-align': 'center', 'outline': 'none', 'background-color': 'rgb(220, 220, 220)'}),
                    width=3),
                    dbc.Col(
                        dcc.Markdown("liters"),
                    width=2),
                    dbc.Col(width=2) 
                ]),
                html.Br(),
                
                dbc.Row([
                    dbc.Col(
                        html.Div(id='checkbox5-output'),
                    width=1),
                    dbc.Col(
                        dcc.Markdown("#### &nbsp;What about horsepower?"),
                    width=6),
                    dbc.Col(
                        dcc.Input(id='hp-input',
                                  placeholder="200 or more?",
                                  style={'border': 'none', 'text-align': 'center', 'outline': 'none', 'background-color': 'rgb(220, 220, 220)'}),
                    width=3), 
                    dbc.Col(
                        dcc.Markdown("hp"),
                    width=2),
                ]),
                html.Br(),
                
                dbc.Row([
                    dbc.Col(
                        html.Div(id='checkbox6-output'),
                    width=1),
                    dbc.Col(
                        dcc.Markdown("#### &nbsp;The gearbox is "),
                    width=4),
                    dbc.Col([
                        dbc.ButtonGroup(id="gearbox-buttons", size="md", className="mb-3", children=[
                            dbc.Button("Automatic", id="gearbox1", color="primary", className="mr-1", n_clicks=0),
                            dbc.Button("Manual", id="gearbox2", color="primary", className="mr-1", n_clicks=0),
                        ]),
                        html.Div(id="gearbox-output") 
                    ], width=6),
                ]),
                
                dbc.Row([
                    dbc.Col(
                        html.Div(id='checkbox7-output'),
                    width=1),
                    dbc.Col(
                        dcc.Markdown("#### &nbsp;What about the fuel?"), 
                    width=5), 
                    dbc.Col(width=6)
                ]),
                
                dbc.Row([
                    dbc.Col(width=1),
                    dbc.Col([
                    dbc.ButtonGroup(id="radio-buttons", size="md", className="mb-3", children=[
                        dbc.Button("Petrol", id="option-petrol", color="primary", className="mr-1", n_clicks=0),
                        dbc.Button("Diesel", id="option-diesel", color="primary", className="mr-1", n_clicks=0),
                        dbc.Button("Hybrid", id="option-hybrid", color="primary", className="mr-1", n_clicks=0),
                        dbc.Button("Hybrid Plug-In", id="option-plugin", color="primary", className="mr-1", n_clicks=0),
                        dbc.Button("Electric", id="option-electric", color="primary", className="mr-1", n_clicks=0),
                        dbc.Button("Petrol+LPG", id="option-lpg", color="primary", className="mr-1", n_clicks=0),
                    ]),
                    html.Div(id="fuel-output")
                    ], width=11),
                ]),
                dbc.Row([
                    dbc.Col(width=4),
                    dbc.Col(
                        html.H1("If you've checked all the boxes, click: ", style = {'font-size': '18px'}),
                    width=6),
                    dbc.Col([
                        dbc.Button("Submit!", id='submit-button', size='md', n_clicks=0), 
                    ],width=2),
                ]),

                
                
            ]), style={'margin': '30px', 'padding': '30px', 'height': '73vh'}, 
                       
            ), width=7),
            dbc.Col([
                html.Br(),
                html.H1("How it works?", style = {'font-size': '35px'}),
                dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            html.P("""To be able to effectively predict the value of a car, it is important to work with up-to-date data. 
                                   To do this, we use current sales listings from a popular online service. 
                                   To view the ads yourself, click the button below:"""),
                            html.A(dbc.Button("Click here"), href="https://www.otomoto.pl/", target="_blank"),
                        ],
                        title="Accessing the data", className="border-dark",

                    ),
                    dbc.AccordionItem(
                        [
                            html.P("""Car Value Predictor is based on a regression model implemented using Python 
                                     and its libraries, mainly selenium and scikit-learn."""),
                        ],
                        title="Model preparation", className="border-dark",
                    ),
                    dbc.AccordionItem(
                        """The user-provided features of the car are transformed accordingly and then 
                        substituted into the trained model to predict a value.""",
                        title="Prediction of value", className="border-dark",
                    ),
                ],always_open=True,),
                html.Br(),
                html.Div(id='predicted-data', style={"fontSize": "30px"}),
                html.Div(id="predicted-value", style={"fontSize": "50px"}),
                html.Br(),

            ]),
        ]),
    html.Div(id='output-data', style={'display': 'none'}), 
    html.Div(id="scrapped-data", style={'display': 'none'}), 
    ],
    style={"padding": "20px", "background-color": "rgb(220, 220, 220)", 'margin': '25px', 'height': '95vh'},
    )],
style={'height': '100vh', 'overflowY': 'hidden' }) 

    
    
@app.callback(
    Output('checkbox1-output', 'children'),
    [Input('makes-dropdown', 'value')]
)
def update_checkbox(make):
    return dcc.Checklist(
        id='checkbox',
        options=[{'label': '', 'value': 'checked'}],
        value=['checked'] if make else []
    )


@app.callback(
    Output('checkbox2-output', 'children'),
    [Input('model-input', 'value')]
)
def update_checkbox2(model):
    return dcc.Checklist(
        id='checkbox2',
        options=[{'label': '', 'value': 'checked'}],
        value=['checked'] if model else [])


@app.callback(
    Output('checkbox-year-output', 'children'),
    [Input('year-input', 'value')]
)
def update_checkbox_year(year):
    return dcc.Checklist(
        id='checkbox-year',
        options=[{'label': '', 'value': 'checked'}],
        value=['checked'] if year else [])


@app.callback(
    Output('checkbox3-output', 'children'),
    [Input('mileage-input', 'value')]
)
def update_checkbox3(mileage):
    return dcc.Checklist(
        id='checkbox3',
        options=[{'label': '', 'value': 'checked'}],
        value=['checked'] if mileage else [])


@app.callback(
    Output('checkbox4-output', 'children'),
    [Input('capacity-input', 'value')]
)
def update_checkbox4(capacity):
    return dcc.Checklist(
        id='checkbox4',
        options=[{'label': '', 'value': 'checked'}],
        value=['checked'] if capacity else [])


@app.callback(
    Output('checkbox5-output', 'children'),
    [Input('hp-input', 'value')]
)
def update_checkbox5(hp):
    return dcc.Checklist(
        id='checkbox5',
        options=[{'label': '', 'value': 'checked'}],
        value=['checked'] if hp else [])


@app.callback(
    Output('checkbox6-output', 'children'),
    [Input('gearbox1', 'n_clicks'),
     Input('gearbox2', 'n_clicks')]
)
def update_checkbox6(n_clicks_gearbox1, n_clicks_gearbox2):
    gearbox_clicked = n_clicks_gearbox1 > 0 or n_clicks_gearbox2 > 0
    
    return dcc.Checklist(
        id='checkbox6',
        options=[{'label': '', 'value': 'checked'}],
        value=['checked'] if gearbox_clicked else []
    )



@app.callback(
    Output('checkbox7-output', 'children'),
    [Input('option-petrol', 'n_clicks'),
     Input('option-diesel', 'n_clicks'),
     Input('option-hybrid', 'n_clicks'),
     Input('option-plugin', 'n_clicks'),
     Input('option-electric', 'n_clicks'),
     Input('option-lpg', 'n_clicks')]
)
def update_checkbox7(n_clicks_petrol, n_clicks_diesel, n_clicks_hybrid, n_clicks_plugin, n_clicks_electric, n_clicks_lpg):
    fuel_clicked = n_clicks_petrol > 0 or n_clicks_diesel > 0 or n_clicks_hybrid > 0 or n_clicks_plugin > 0 or n_clicks_electric > 0 or n_clicks_lpg > 0 
    
    return dcc.Checklist(
        id='checkbox7',
        options=[{'label': '', 'value': 'checked'}],
        value=['checked'] if fuel_clicked else [])





@app.callback(
    Output('year-input', 'value'),
    [Input('year-input', 'value')])

def format_year(value):
    if value is not None:
        value = re.sub(r'\D', '', value)
        if len(value) > 4:
            value = value[:4]
    return value

    

@app.callback(
    Output('mileage-input', 'value'),
    [Input('mileage-input', 'value')])

def format_mileage(value):
    if value is not None:
        value = re.sub(r'\D', '', value)
        if value.isdigit():
            value = '{:,}'.format(int(value))
    return value



@app.callback(
    Output('hp-input', 'value'),
    [Input('hp-input', 'value')])

def format_mileage2(value2):
    if value2 is not None:
        value2 = re.sub(r'\D', '', value2)
        if value2.isdigit():
            value2 = '{:,}'.format(int(value2))
    return value2





@app.callback(
    [Output('gearbox1', 'color'),
     Output('gearbox2', 'color')],
    [Input('gearbox1', 'n_clicks'),
     Input('gearbox2', 'n_clicks')]
)
    
def update_gearbox_colors(click1, click2):
    ctx = callback_context
    if not ctx.triggered:
        return "primary", "primary"
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        colors = ["primary", "primary"]
        if triggered_id == "gearbox1":
            colors[0] = "success"
        elif triggered_id == "gearbox2":
            colors[1] = "success"
        return tuple(colors)
    

    

@app.callback(
    [Output('option-petrol', 'color'),
     Output('option-diesel', 'color'),
     Output('option-hybrid', 'color'),
     Output('option-plugin', 'color'),
     Output('option-electric', 'color'),
     Output('option-lpg', 'color')],
    [Input('option-petrol', 'n_clicks'),
     Input('option-diesel', 'n_clicks'),
     Input('option-hybrid', 'n_clicks'),
     Input('option-plugin', 'n_clicks'),
     Input('option-electric', 'n_clicks'),
     Input('option-lpg', 'n_clicks')]
)
    
def update_fuel_colors(click1, click2, click3, click4, click5, click6):
    ctx = callback_context
    if not ctx.triggered:
        return "primary", "primary", "primary", "primary", "primary", "primary"
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        colors = ["primary", "primary", "primary", "primary", "primary", "primary"]
        if triggered_id == "option-petrol":
            colors[0] = "success"
        elif triggered_id == "option-diesel":
            colors[1] = "success"
        elif triggered_id == "option-hybrid":
            colors[2] = "success"
        elif triggered_id == "option-plugin":
            colors[3] = "success"
        elif triggered_id == "option-electric":
            colors[4] = "success"
        elif triggered_id == "option-lpg":
            colors[5] = "success"
        return tuple(colors)
    
    
    
@app.callback(
    Output('output-data', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('makes-dropdown', 'value'),
     State('model-input', 'value'),
     State('year-input', 'value'),
     State('mileage-input', 'value'),
     State('capacity-input', 'value'),
     State('hp-input', 'value'),
     State('gearbox1', 'n_clicks'),
     State('gearbox2', 'n_clicks'),
     State('option-petrol', 'n_clicks'),
     State('option-diesel', 'n_clicks'),
     State('option-hybrid', 'n_clicks'),
     State('option-plugin', 'n_clicks'),
     State('option-electric', 'n_clicks'),
     State('option-lpg', 'n_clicks')]
)
def update_input_data(n_clicks, make, model, year, mileage, capacity, horsepower, gearbox1, gearbox2, petrol, diesel,
                    hybrid, plugin, electric, lpg):
    global input_data
    if n_clicks:
        gearbox = 'Automatic' if gearbox1 else 'Manual'
        fuel = ''
        if petrol:
            fuel = 'Petrol'
        elif diesel:
            fuel = 'Diesel'
        elif hybrid:
            fuel = 'Hybrid'
        elif plugin:
            fuel = 'Hybrid Plug-In'
        elif electric:
            fuel = 'Electric'
        elif lpg:
            fuel = 'Petrol+LPG'

        input_data = pd.DataFrame(columns=['Make', 'Model', 'Year', 'Mileage', 'Capacity', 'Horsepower', 'Gearbox', 'Fuel'])
        input_data.loc[len(input_data)] = {'Make': make, 'Model': model, 'Year': year, 'Mileage': mileage, 'Capacity': capacity,
                                            'Horsepower': horsepower, 'Gearbox': gearbox, 'Fuel': fuel}

        input_data = input_data.to_json(orient='records')
        
        return input_data


        
        
@app.callback(
    Output("scrapped-data", "children"),
    [Input('submit-button', 'n_clicks')],
    [State("makes-dropdown", "value"), State("model-input", "value"), State("year-input", "value")]
)


def scrape_data(n_clicks, make, model, year):
    
    if n_clicks > 0:
        
        url = 'https://www.otomoto.pl'
        driver = webdriver.Safari()
        driver.get(url)

        #cookies
        element0 = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="onetrust-accept-btn-handler"]')))
        element0.click()

        #make
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/div/div/div/main/div[2]/article/article/fieldset/form/section[1]/div[1]/div[1]/div/input'))
        ).click()
        make2 = driver.find_element(By.XPATH, f'//*[@id="{make.lower()}"]')
        make2.click()
        time.sleep(1)

        #model
        model1 = driver.find_element(By.XPATH, '//*[@id="__next"]/div/div/div/main/div[2]/article/article/fieldset/form/section[1]/div[2]/div/div/input')
        model1.click()

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, f'//*[@id="{model.lower()}"]'))
        ).click()
        time.sleep(1)

        #year
        year1 = driver.find_element(By.XPATH, '//*[@id="__next"]/div/div/div/main/div[2]/article/article/fieldset/form/section[1]/div[4]/div/div/input')
        year1.click()
        time.sleep(1)
        year2 = driver.find_element(By.XPATH, f'//*[@id="{year}"]')
        year2.click()
        time.sleep(1)

        #show ads
        element3 = driver.find_element(By.XPATH, '//*[@id="__next"]/div/div/div/main/div[2]/article/article/fieldset/form/section[2]/button[1]/span/span')
        element3.click()

        #prices
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/div/div/div/div[2]/main/div[2]/div/div[3]/div[2]/div[1]/article/section'))
        )
        price_elements = driver.find_elements(By.XPATH, '//*[@id="__next"]/div/div/div/div[2]/main/div[2]/div/div[3]/div[2]/div/article/section/div[4]/div[2]/div[1]/h3')
        prices = [price.text for price in price_elements]

        #mileage
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/div/div/div/div[2]/main/div[2]/div/div[3]/div[2]/div[1]/article/section'))
        )
        mileage_elements = driver.find_elements(By.XPATH, '//*[@id="__next"]/div/div/div/div[2]/main/div[2]/div/div[3]/div[2]/div/article/section/div[3]/dl[1]/dd[1]')
        mileage = [km.text for km in mileage_elements]

        #fuel
        fuel_elements = driver.find_elements(By.XPATH, '//*[@id="__next"]/div/div/div/div[2]/main/div[2]/div/div[3]/div[2]/div/article/section/div[3]/dl[1]/dd[2]')
        fuels = [fuel.text for fuel in fuel_elements]

        #gearbox
        gears_elements = driver.find_elements(By.XPATH, '//*[@id="__next"]/div/div/div/div[2]/main/div[2]/div/div[3]/div[2]/div/article/section/div[3]/dl[1]/dd[3]')
        gears = [gear.text for gear in gears_elements]

        #year2
        years_elements = driver.find_elements(By.XPATH, '//*[@id="__next"]/div/div/div/div[2]/main/div[2]/div/div[3]/div[2]/div/article/section/div[3]/dl[1]/dd[4]')
        years = [year.text for year in years_elements]

        #info
        info_elements = driver.find_elements(By.XPATH, '//*[@id="__next"]/div/div/div/div[2]/main/div[2]/div/div[3]/div[2]/div/article/section/div[2]/p')
        infos = [info.text for info in info_elements]
        capacity = []
        horsepower = []
        for i in infos:
            elements = i.split('•')
            elements = [element.strip() for element in elements]

            capacity.append(elements[0])
            horsepower.append(elements[1])

        data = zip_longest(years, mileage, capacity, horsepower, gears, fuels, prices)
        raw_df = pd.DataFrame(data, columns=['Year', 'Mileage', 'Capacity', 'Horsepower', 'Gearbox', 'Fuel', 'Price'])

        #driver.quit()

        num_cols = ['Year', 'Mileage', 'Horsepower', 'Price']
        df = raw_df.copy()

        for col in num_cols:
            df[col] = df[col].astype(str).str.replace(r'\D', '', regex=True) #remove all non-digit characters
            df[col] = pd.to_numeric(df[col], errors='coerce') #if cannot be converted -> NaN

        df['Capacity'] = df['Capacity'].str.replace(' cm3', '', regex=False)
        df['Capacity'] = df['Capacity'].str.replace(r'\s', '', regex=True) #regex -> regular expressions like r'\D' or r'\s'
        df['Capacity'] = df['Capacity'].astype(int)
        df['Capacity'] = (df['Capacity'] / 1000).round(1)

        df = df.dropna()
        df['Mileage'] = df['Mileage'].astype(int)
        df['Year'] = df['Year'].astype(int)
        df = pd.concat([df] * 10000, ignore_index=True)
        

        df = df.to_json(orient='records')
        return df





        
@app.callback(
    [Output("predicted-data", "children"),
     Output("predicted-value", "children")],
    [Input('scrapped-data', 'children')],
    [State('makes-dropdown', 'value'),
     State('model-input', 'value'),
     State('year-input', 'value'),
     State('mileage-input', 'value'),
     State('capacity-input', 'value'),
     State('hp-input', 'value'),
     State('gearbox1', 'n_clicks'),
     State('gearbox2', 'n_clicks'),
     State('option-petrol', 'n_clicks'),
     State('option-diesel', 'n_clicks'),
     State('option-hybrid', 'n_clicks'),
     State('option-plugin', 'n_clicks'),
     State('option-electric', 'n_clicks'),
     State('option-lpg', 'n_clicks')]
)


def predicted_data(scrapped_data, make, model, year, mileage, capacity, hp, gearbox1, gearbox2, petrol, diesel, hybrid, plugin, electric, lpg):
    
    
    if not scrapped_data:
        return [], []
    else:
        predict_data = pd.DataFrame({'Make': make,
                                       'Model': model,
                                       'Year': year,
                                       'Mileage': mileage,
                                       'Capacity': capacity,
                                       'Horsepower': hp,
                                       'Gearbox_Automatyczna': gearbox1,
                                       'Gearbox_Manualna': gearbox2,
                                       'Fuel_Benzyna': petrol,
                                       'Fuel_Diesel': diesel,
                                       'Fuel_Hybryda': hybrid,
                                       'Fuel_Hybryda Plug-in': plugin,
                                       'Fuel_Elektryczny': electric,
                                       'Fuel_Benzyna+LPG': lpg}, index=[14])
        
        
        predict_data['Mileage'] = predict_data['Mileage'].astype(str).str.replace(r'\D', '', regex=True)      
        

        df = pd.read_json(scrapped_data)
        
        target = df.pop('Price')
        data = df
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.15)
        
        
        scaler = StandardScaler() 
        num_cols2 = list(X_train.select_dtypes(include='number').columns)

        X_train[num_cols2] = scaler.fit_transform(X_train[num_cols2]) 
        X_train = pd.get_dummies(data=X_train, drop_first=True, prefix_sep='_')
        
        X_test[num_cols2] = scaler.transform(X_test[num_cols2]) 
        X_test = pd.get_dummies(data=X_test, drop_first=True, prefix_sep='_')
        
        
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test) 
        y_pred = np.abs(y_pred)
        
        col_xtrain = X_train.columns.tolist()
        predict_data = predict_data[col_xtrain]
        
        predict_data[['Year', 'Mileage', 'Capacity', 'Horsepower']] = scaler.transform(predict_data[['Year', 'Mileage', 'Capacity', 'Horsepower']]) 

        predicted_price = regressor.predict(predict_data)
        predicted_price = np.abs(predicted_price)[0]
        
        text = f"Predicted value of your {make} {model} is"
        value = f"{predicted_price:,.0f} PLN"

        return html.Div(text), html.Div(value)

    
    
    
    
   

def open_browser():
    webbrowser.open_new_tab('http://127.0.0.1:8787')

if __name__ == '__main__':
    app.title = "Car Value Predictor" 
    import threading
    threading.Timer(1, open_browser).start()
    app.run_server(port=8787)
