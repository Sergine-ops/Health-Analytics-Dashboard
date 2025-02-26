import dash
from dash import dcc, html, Input, Output, State, callback_context
from scipy.interpolate import griddata 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import uuid
import json
import random


# DATA GENERATION

def generate_synthetic_health_data(num_records=1000):
    """
    Generate a synthetic public health dataset with various metrics
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define regions, age groups, and conditions
    regions = ['Northeast', 'Midwest', 'South', 'West', 'Northwest', 'Southeast', 'Southwest', 'Central']
    districts = {
        'Northeast': ['NE-A', 'NE-B', 'NE-C', 'NE-D'],
        'Midwest': ['MW-A', 'MW-B', 'MW-C'],
        'South': ['S-A', 'S-B', 'S-C', 'S-D', 'S-E'],
        'West': ['W-A', 'W-B', 'W-C'],
        'Northwest': ['NW-A', 'NW-B'],
        'Southeast': ['SE-A', 'SE-B', 'SE-C'],
        'Southwest': ['SW-A', 'SW-B', 'SW-C'],
        'Central': ['C-A', 'C-B', 'C-C', 'C-D']
    }
    
    age_groups = ['0-17', '18-29', '30-44', '45-64', '65+']
    medical_conditions = ['Cardiovascular', 'Respiratory', 'Diabetes', 'Hypertension', 'Mental Health', 'Cancer', 'Obesity']
    hospital_types = ['Public', 'Private', 'University', 'Community']
    
    # Generate dates spanning 3 years
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    
    # Initialize empty lists for each column
    data = {
        'record_id': [],
        'date': [],
        'region': [],
        'district': [],
        'age_group': [],
        'condition': [],
        'hospital_type': [],
        'cases': [],
        'hospitalizations': [],
        'icu_admissions': [],
        'recovery_days': [],
        'readmission_rate': [],
        'treatment_cost': [],
        'vaccination_rate': [],
        'satisfaction_score': [],
        'population_density': [],
        'healthcare_access_index': [],
        'air_quality_index': []
    }
    
    # Base region stats to ensure regional patterns
    region_stats = {
        region: {
            'base_cases': np.random.uniform(50, 150),
            'base_hosp': np.random.uniform(10, 40),
            'base_icu': np.random.uniform(2, 10),
            'base_vax': np.random.uniform(0.5, 0.9),
            'base_cost': np.random.uniform(5000, 15000),
            'pop_density': np.random.uniform(50, 300),
            'healthcare_access': np.random.uniform(0.3, 0.9),
            'air_quality': np.random.uniform(30, 90)
        } for region in regions
    }
    
    # Condition-specific multipliers
    condition_multipliers = {
        'Cardiovascular': {'cases': 1.2, 'hosp': 1.5, 'icu': 2.0, 'cost': 1.8},
        'Respiratory': {'cases': 1.5, 'hosp': 1.3, 'icu': 1.7, 'cost': 1.2},
        'Diabetes': {'cases': 1.0, 'hosp': 1.1, 'icu': 1.2, 'cost': 1.3},
        'Hypertension': {'cases': 1.3, 'hosp': 1.0, 'icu': 0.9, 'cost': 1.0},
        'Mental Health': {'cases': 1.8, 'hosp': 0.7, 'icu': 0.5, 'cost': 1.1},
        'Cancer': {'cases': 0.5, 'hosp': 1.8, 'icu': 1.5, 'cost': 2.5},
        'Obesity': {'cases': 1.4, 'hosp': 0.8, 'icu': 0.7, 'cost': 0.9}
    }
    
    # Age group risk factors
    age_risk_factors = {
        '0-17': {'cases': 0.7, 'hosp': 0.5, 'icu': 0.3},
        '18-29': {'cases': 0.9, 'hosp': 0.6, 'icu': 0.4},
        '30-44': {'cases': 1.0, 'hosp': 0.8, 'icu': 0.7},
        '45-64': {'cases': 1.3, 'hosp': 1.2, 'icu': 1.3},
        '65+': {'cases': 1.5, 'hosp': 1.8, 'icu': 2.2}
    }
    
    # Season effect (by month)
    seasonal_effects = {
        1: 1.4, 
        2: 1.3,
        3: 1.1,
        4:.9,
        5: 0.8,
        6: 0.7,  
        7: 0.7,
        8: 0.8,
        9: 0.9,
        10: 1.0,
        11: 1.1,
        12: 1.3   
    }
    
    # Year-over-year improvement trend (healthcare is getting better)
    yoy_improvement = {
        2021: 1.0,
        2022: 0.92,
        2023: 0.85
    }
    
    # Generate records
    for _ in range(num_records):
        # Random date from range
        days_offset = np.random.randint(0, date_range)
        record_date = start_date + timedelta(days=days_offset)
        
        # Select demographic attributes
        region = np.random.choice(regions)
        district = np.random.choice(districts[region])
        age_group = np.random.choice(age_groups, p=[0.2, 0.2, 0.2, 0.2, 0.2])
        condition = np.random.choice(medical_conditions)
        hospital_type = np.random.choice(hospital_types)
        
        # Get base metrics for region
        base_cases = region_stats[region]['base_cases']
        base_hosp = region_stats[region]['base_hosp']
        base_icu = region_stats[region]['base_icu']
        base_vax = region_stats[region]['base_vax']
        base_cost = region_stats[region]['base_cost']
        
        # Apply multipliers based on condition, age, season, and year
        condition_factor = condition_multipliers[condition]
        age_factor = age_risk_factors[age_group]
        season_factor = seasonal_effects[record_date.month]
        year_factor = yoy_improvement[record_date.year]
        
        # Small random noise factor for variation
        noise = np.random.normal(1, 0.15)
        
        # Calculate metrics
        cases = int(base_cases * condition_factor['cases'] * age_factor['cases'] * season_factor * year_factor * noise)
        hospitalizations = int(base_hosp * condition_factor['hosp'] * age_factor['hosp'] * season_factor * year_factor * noise)
        icu_admissions = int(base_icu * condition_factor['icu'] * age_factor['icu'] * season_factor * year_factor * noise)
        
        # Ensure logical relationships
        hospitalizations = min(hospitalizations, cases)
        icu_admissions = min(icu_admissions, hospitalizations)
        
        # Additional metrics
        recovery_days = np.random.lognormal(mean=np.log(10), sigma=0.5) * condition_factor.get('cost', 1) * age_factor.get('hosp', 1)
        readmission_rate = min(0.3, max(0.01, np.random.beta(2, 8) * condition_factor.get('hosp', 1)))
        treatment_cost = base_cost * condition_factor['cost'] * np.random.lognormal(mean=0, sigma=0.2)
        
        # Vaccination rate with regional and temporal patterns
        vax_rate = min(0.95, max(0.3, base_vax * (1 + (record_date.year - 2021) * 0.1) * np.random.normal(1, 0.05)))
        
        # Patient satisfaction (higher in some regions, hospital types)
        satisfaction_bonus = 0.1 if hospital_type in ['University', 'Private'] else 0
        satisfaction_score = min(5, max(1, np.random.normal(3.7, 0.5) + satisfaction_bonus))
        
        # Environmental and socioeconomic factors
        population_density = region_stats[region]['pop_density'] * np.random.normal(1, 0.1)
        healthcare_access = region_stats[region]['healthcare_access'] * np.random.normal(1, 0.1)
        air_quality = region_stats[region]['air_quality'] * np.random.normal(1, 0.15)
        
        # Add to data dictionary
        data['record_id'].append(str(uuid.uuid4())[:8])
        data['date'].append(record_date)
        data['region'].append(region)
        data['district'].append(district)
        data['age_group'].append(age_group)
        data['condition'].append(condition)
        data['hospital_type'].append(hospital_type)
        data['cases'].append(cases)
        data['hospitalizations'].append(hospitalizations)
        data['icu_admissions'].append(icu_admissions)
        data['recovery_days'].append(round(recovery_days, 1))
        data['readmission_rate'].append(round(readmission_rate, 3))
        data['treatment_cost'].append(round(treatment_cost, 2))
        data['vaccination_rate'].append(round(vax_rate, 3))
        data['satisfaction_score'].append(round(satisfaction_score, 1))
        data['population_density'].append(round(population_density, 1))
        data['healthcare_access_index'].append(round(healthcare_access, 2))
        data['air_quality_index'].append(round(air_quality, 1))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    #  calculated columns
    df['mortality_rate'] = (df['icu_admissions'] / df['cases'] * np.random.normal(0.4, 0.1)).clip(0.01, 0.3)
    df['cost_per_case'] = df['treatment_cost'] / np.maximum(1, df['cases'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['month_name'] = df['date'].dt.strftime('%b')
    df['day_of_week'] = df['date'].dt.day_name()
    
    # a geographic coordinate for each district 
    district_coords = {}
    for region in regions:
        base_lat = np.random.uniform(30, 45)
        base_lon = np.random.uniform(-120, -75)
        
        for district in districts[region]:
            district_coords[district] = {
                'latitude': base_lat + np.random.uniform(-2, 2),
                'longitude': base_lon + np.random.uniform(-2, 2)
            }
    
    #  coordinates to dataframe
    df['latitude'] = df['district'].map(lambda x: district_coords[x]['latitude'])
    df['longitude'] = df['district'].map(lambda x: district_coords[x]['longitude'])
    
    return df

# Generate our dataset
df = generate_synthetic_health_data(1500)

# INITIALIZE DASH APP

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css"
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True
)

app.title = "Public Health Analytics Dashboard"
server = app.server


# REUSABLE COMPONENTS


def create_card(title, value, change, icon_class="fas fa-chart-line"):
    change_color = "text-emerald-400" if float(change) >= 0 else "text-rose-400"
    change_arrow = "↑" if float(change) >= 0 else "↓"
    
    return html.Div(
        className="bg-gray-800 rounded-lg shadow-lg p-4 border border-gray-700",
        children=[
            html.Div(className="flex items-center justify-between", children=[
                html.H3(title, className="text-gray-300 text-sm font-medium"),
                html.Span(className=f"text-xl {icon_class} text-gray-400")
            ]),
            html.Div(className="mt-2", children=[
                html.H2(value, className="text-2xl font-bold text-white"),
                html.Span(
                    f"{change_arrow} {abs(float(change))}%", 
                    className=f"text-xs {change_color} font-semibold"
                )
            ])
        ]
    )
def create_filter_bar():
    return html.Div(
        className="bg-gray-800 rounded-lg shadow-lg p-4 mb-4 border border-gray-700",
        children=[
            html.Div(
                className="grid grid-cols-1 md:grid-cols-5 gap-4",
                children=[
                    html.Div([
                        html.Label("Date Range", className="block text-sm font-medium text-gray-300"),
                        dcc.DatePickerRange(
                            id='date-range-filter',
                            # Remove df references for this example, add your data later
                            className="w-full bg-gray-700 text-white border-gray-600"
                        )
                    ]),
                    html.Div([
                        html.Label("Region", className="block text-sm font-medium text-gray-300"),
                        dcc.Dropdown(
                            id='region-filter',
                            multi=True,
                            placeholder="All Regions",
                            className="w-full bg-gray-700 text-white border-gray-600"
                        )
                    ]),
                    html.Div([
                        html.Label("Condition", className="block text-sm font-medium text-gray-300"),
                        dcc.Dropdown(
                            id='condition-filter',
                            multi=True,
                            placeholder="All Conditions",
                            className="w-full bg-gray-700 text-white border-gray-600"
                        )
                    ]),
                    html.Div([
                        html.Label("Age Group", className="block text-sm font-medium text-gray-300"),
                        dcc.Dropdown(
                            id='age-filter',
                            multi=True,
                            placeholder="All Age Groups",
                            className="w-full bg-gray-700 text-white border-gray-600"
                        )
                    ]),
                    html.Div([
                        html.Label("Hospital Type", className="block text-sm font-medium text-gray-300"),
                        dcc.Dropdown(
                            id='hospital-filter',
                            multi=True,
                            placeholder="All Hospital Types",
                            className="w-full bg-gray-700 text-white border-gray-600"
                        )
                    ])
                ]
            ),
            html.Div(
                className="flex justify-end mt-4",
                children=[
                    html.Button(
                        "Apply Filters",
                        id="apply-filters-button",
                        className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 px-4 rounded-md"
                    ),
                    html.Button(
                        "Reset Filters",
                        id="reset-filters-button",
                        className="ml-2 bg-gray-600 hover:bg-gray-500 text-white font-medium py-2 px-4 rounded-md"
                    )
                ]
            )
        ]
    )

def create_navbar(pathname):
    """Create navigation bar"""

    # Base classes for inactive links
    base_classes = "px-3 py-2 rounded-md text-lg font-medium text-gray-300 hover:bg-indigo-700 hover:text-white"
    # Active classes
    active_classes = "px-3 py-2 rounded-md text-lg font-medium bg-indigo-800 text-white"

    pathname = pathname or "/"

    return html.Nav(
        className="bg-white shadow-md px-6 py-8 mb-6 h-24 sticky top-0 z-50",
        children=[
            html.Div(
                className="container mx-auto flex flex-wrap items-center justify-between",
                children=[
                    html.A(
                        className="flex items-center text-blue-600",
                        href="/",
                        children=[
                            html.Span("🏥", className="text-2xl mr-2"),
                            html.Span("Public Health Analytics", className="font-bold text-2xl")
                        ]
                    ),
                    html.Div(
                        className="flex w-full md:w-auto md:flex-grow-0 mt-4 md:mt-0 justify-center",
                        children=[
                            html.Ul(
                                className="flex flex-wrap md:flex-nowrap space-x-0 md:space-x-2 space-y-2 md:space-y-0 justify-center",
                                children=[
                                    html.Li(
                                        dcc.Link(
                                            "Overview",
                                            href="/",
                                            className=active_classes if pathname == "/" else base_classes,
                                            id="nav-overview"
                                        )
                                    ),
                                    html.Li(
                                        dcc.Link(
                                            "Trends",
                                            href="/trends",
                                            className=active_classes if pathname == "/trends" else base_classes,                                            id="nav-trends"
                                        )
                                    ),
                                    html.Li(
                                        dcc.Link(
                                            "Comparisons",
                                            href="/comparisons",
                                            className=active_classes if pathname == "/comparisons" else base_classes,
                                            id="nav-comparisons"
                                        )
                                    ),
                                    html.Li(
                                        dcc.Link(
                                            "Geospatial",
                                            href="/geospatial",
                                            className=active_classes if pathname == "/geospatial" else base_classes,
                                            id="nav-geospatial"
                                        )
                                    ),
                                    html.Li(
                                        dcc.Link(
                                            "Predictions",
                                            href="/predictions",
                                            className=active_classes if pathname == "/predictions" else base_classes,
                                            id="nav-predictions"
                                        )
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="flex items-center",
                        children=[
                            html.Button(
                                "Export Data",
                                id="export-button",
                                className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-3 py-1 text-sm rounded hidden md:block"
                            )
                        ]
                    )
                ]
            )
        ]
    )

# PAGE LAYOUTS

# 1. OVERVIEW PAGE
overview_layout = lambda pathname="/": html.Div([
    html.Div(
        className="container mx-auto px-4",
        children=[
            # Page Header
            html.Div(
                className="mb-6",
                children=[
                    html.H1("Public Health Overview", className="text-2xl font-bold text-gray-800"),
                    html.P(
                        "Comprehensive view of key health metrics and trends across regions and conditions",
                        className="text-gray-600"
                    )
                ]
            ),
            # Filters Section
            create_filter_bar(),
            # Key Metrics Cards
            html.Div(
                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6",
                id="overview-metric-cards",
                children=[
                    create_card("Total Cases", "42,546", "+5.2", "fas fa-procedures"),
                    create_card("Hospitalizations", "12,385", "-2.1", "fas fa-hospital"),
                    create_card("Avg. Recovery Days", "14.2", "+0.8", "fas fa-calendar"),
                    create_card("Mortality Rate", "1.8%", "-0.3", "fas fa-heartbeat")
                ]
            ),
            # Main Charts Row
            html.Div(
                className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6",
                children=[
                    # Trend Chart
                    html.Div(
                        className="bg-white rounded-lg shadow-md p-4 border border-gray-200",
                        children=[
                            html.H3("Case Trends by Condition", className="text-lg font-semibold mb-4"),
                            dcc.Graph(id="overview-trend-chart", style={"height": "350px"})
                        ]
                    ),
                    # Distribution Chart
                    html.Div(
                        className="bg-white rounded-lg shadow-md p-4 border border-gray-200",
                        children=[
                            html.H3("Case Distribution by Region", className="text-lg font-semibold mb-4"),
                            dcc.Graph(id="overview-distribution-chart", style={"height": "350px"})
                        ]
                    )
                ]
            ),
            # Second Row of Charts
            html.Div(
                className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6",
                children=[
                    # Hospitalization vs Cases
                    html.Div(
                        className="bg-white rounded-lg shadow-md p-4 border border-gray-200",
                        children=[
                            html.H3("Hospitalization to Case Ratio", className="text-lg font-semibold mb-4"),
                            dcc.Graph(id="overview-hosp-ratio-chart", style={"height": "300px"})
                        ]
                    ),
                    # Age Group Breakdown
                    html.Div(
                        className="bg-white rounded-lg shadow-md p-4 border border-gray-200",
                        children=[
                            html.H3("Case Distribution by Age Group", className="text-lg font-semibold mb-4"),
                            dcc.Graph(id="overview-age-chart", style={"height": "300px"})
                        ]
                    ),
                    # Hospital Type Performance
                    html.Div(
                        className="bg-white rounded-lg shadow-md p-4 border border-gray-200",
                        children=[
                            html.H3("Metrics by Hospital Type", className="text-lg font-semibold mb-4"),
                            dcc.Graph(id="overview-hospital-chart", style={"height": "300px"})
                        ]
                    )
                ]
            ),
            # Third Row - Heat Calendar
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("Seasonal Patterns of Cases", className="text-lg font-semibold mb-4"),
                    dcc.Graph(id="overview-calendar-chart", style={"height": "400px"})
                ]
            ),
            # Additional Info Section
            html.Div(
                className="bg-gray-50 rounded-lg p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("About This Dashboard", className="text-lg font-semibold mb-2"),
                    html.P(
                        "This comprehensive public health analytics dashboard provides insights into case trends, " +
                        "hospitalizations, and health outcomes across various regions and demographics. Use the filters " +
                        "above to customize your view and explore different dimensions of the data.",
                        className="text-gray-600 mb-2"
                    ),
                    html.P(
                        "The data is updated daily from regional health authorities and hospitals across the country. " +
                        "All metrics are calculated based on standardized definitions for consistency and comparability.",
                        className="text-gray-600"
                    )
                ]
            )
        ]
    )
])

# 2. TRENDS PAGE
trends_layout = lambda pathname="/": html.Div([
    html.Div(
        className="container mx-auto px-4",
        children=[
            # Page Header
            html.Div(
                className="mb-6",
                children=[
                    html.H1("Temporal Trends Analysis", className="text-2xl font-bold text-gray-800"),
                    html.P(
                        "Analyzing patterns and changes in health metrics over time",
                        className="text-gray-600"
                    )
                ]
            ),
            # Filters Section
            create_filter_bar(),
            # Interactive Time Series Chart
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.Div(
                        className="flex flex-wrap items-center justify-between mb-4",
                        children=[
                            html.H3("Multi-Metric Time Series Analysis", className="text-lg font-semibold"),
                            html.Div(
                                className="flex space-x-2 mt-2 sm:mt-0",
                                children=[
                                    dcc.Dropdown(
                                        id="trends-time-grouping",
                                        options=[
                                            {"label": "Daily", "value": "D"},
                                            {"label": "Weekly", "value": "W"},
                                            {"label": "Monthly", "value": "ME"},
                                            {"label": "Quarterly", "value": "Q"}
                                        ],
                                        value="ME",
                                        clearable=False,
                                        className="w-36"
                                    ),
                                    dcc.Dropdown(
                                        id="trends-metrics-selector",
                                        options=[
                                            {"label": "Cases", "value": "cases"},
                                            {"label": "Hospitalizations", "value": "hospitalizations"},
                                            {"label": "ICU Admissions", "value": "icu_admissions"},
                                            {"label": "Recovery Days", "value": "recovery_days"},
                                            {"label": "Treatment Cost", "value": "treatment_cost"},
                                            {"label": "Mortality Rate", "value": "mortality_rate"}
                                        ],
                                        value=["cases", "hospitalizations"],
                                        multi=True,
                                        className="w-64"
                                    )
                                ]
                            )
                        ]
                    ),
                    dcc.Graph(id="trends-time-series-chart", style={"height": "400px"})
                ]
            ),
            # Animated Trend Chart
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.Div(
                        className="flex flex-wrap items-center justify-between mb-4",
                        children=[
                            html.H3("Regional Metrics Evolution", className="text-lg font-semibold"),
                            html.Div(
                                className="flex space-x-2 mt-2 sm:mt-0",
                                children=[
                                    dcc.Dropdown(
                                        id="trends-x-axis",
                                        options=[
                                            {"label": "Vaccination Rate (%)", "value": "vaccination_rate"},
                                            {"label": "Healthcare Access Index", "value": "healthcare_access_index"},
                                            {"label": "Population Density", "value": "population_density"},
                                            {"label": "Treatment Cost ($)", "value": "treatment_cost"}
                                        ],
                                        value="vaccination_rate",
                                        clearable=False,
                                        className="w-48"
                                    ),
                                    dcc.Dropdown(
                                        id="trends-y-axis",
                                        options=[
                                            {"label": "Cases per 100K", "value": "cases"},
                                            {"label": "Hospitalization Rate (%)", "value": "hospitalizations"},
                                            {"label": "ICU Admission Rate (%)", "value": "icu_admissions"},
                                            {"label": "Mortality Rate (%)", "value": "mortality_rate"}
                                        ],
                                        value="mortality_rate",
                                        clearable=False,
                                        className="w-48"
                                    )
                                ]
                            )
                        ]
                    ),
                    dcc.Graph(id="trends-animated-chart", style={"height": "500px"}),
                    html.Div(
                        className="flex justify-center mt-4",
                        children=[
                            html.Button(
                                "▶ Play Animation", 
                                id="trends-animation-button",
                                className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md"
                            )
                        ]
                    )
                ]
            ),
            # Third Row - Dual Analyses
            html.Div(
                className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6",
                children=[
                    # Seasonal Decomposition
                    html.Div(
                        className="bg-white rounded-lg shadow-md p-4 border border-gray-200",
                        children=[
                            html.H3("Seasonal Patterns", className="text-lg font-semibold mb-4"),
                            dcc.Dropdown(
                                id="trends-seasonal-metric",
                                options=[
                                    {"label": "Cases", "value": "cases"},
                                    {"label": "Hospitalizations", "value": "hospitalizations"},
                                    {"label": "ICU Admissions", "value": "icu_admissions"}
                                ],
                                value="cases",
                                clearable=False,
                                className="w-48 mb-4"
                            ),
                            dcc.Graph(id="trends-seasonal-chart", style={"height": "300px"})
                        ]
                    ),
                    html.Div(
                        className="bg-white rounded-lg shadow-md p-4 border border-gray-200",
                        children=[
                            html.H3("Rolling Average Trends", className="text-lg font-semibold mb-4"),
                            dcc.Slider(
                                id="trends-rolling-window",
                                min=7,
                                max=90,
                                step=7,
                                value=30,
                                marks={i: f"{i}d" for i in range(7, 91, 14)},
                                className="mb-4"
                            ),
                            dcc.Graph(id="trends-rolling-chart", style={"height": "300px"})
                        ]
                    )
                ]
            )
        ]
    )
])

# 3. COMPARISONS PAGE
comparisons_layout = lambda pathname="/": html.Div([
    html.Div(
        className="container mx-auto px-4",
        children=[
            html.Div(
                className="mb-6",
                children=[
                    html.H1("Comparative Analysis", className="text-2xl font-bold text-gray-800"),
                    html.P("Compare health metrics across regions, conditions, and demographics", className="text-gray-600")
                ]
            ),
            create_filter_bar(),
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("Metric Comparison by Region", className="text-lg font-semibold mb-4"),
                    dcc.Dropdown(
                        id="comparisons-metric",
                        options=[{"label": m.capitalize(), "value": m} for m in ["cases", "hospitalizations", "treatment_cost", "mortality_rate"]],
                        value="cases",
                        clearable=False,
                        className="w-48 mb-4"
                    ),
                    dcc.Graph(id="comparisons-bar-chart", style={"height": "400px"})
                ]
            ),
            html.Div(
                className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6",
                children=[
                    html.Div(
                        className="bg-white rounded-lg shadow-md p-4 border border-gray-200",
                        children=[
                            html.H3("Distribution by Condition", className="text-lg font-semibold mb-4"),
                            dcc.Graph(id="comparisons-box-chart", style={"height": "350px"})
                        ]
                    ),
                    html.Div(
                        className="bg-white rounded-lg shadow-md p-4 border border-gray-200",
                        children=[
                            html.H3("Correlation Heatmap", className="text-lg font-semibold mb-4"),
                            dcc.Graph(id="comparisons-heatmap", style={"height": "350px"})
                        ]
                    )
                ]
            ),
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("Cases by Region and Condition (Treemap)", className="text-lg font-semibold mb-4"),
                    dcc.Graph(id="comparisons-treemap", style={"height": "400px"})
                ]
            ),
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("Patient Flow (Sankey)", className="text-lg font-semibold mb-4"),
                    dcc.Graph(id="comparisons-sankey", style={"height": "400px"})
                ]
            )
        ]
    )
])
# 4. GEOSPATIAL PAGE
geospatial_layout = lambda pathname="/": html.Div([
    html.Div(
        className="container mx-auto px-4",
        children=[
            html.Div(
                className="mb-6",
                children=[
                    html.H1("Geospatial Insights", className="text-2xl font-bold text-gray-800"),
                    html.P("Visualize health metrics geographically across districts", className="text-gray-600")
                ]
            ),
            create_filter_bar(),
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("District-Level Health Metrics", className="text-lg font-semibold mb-4"),
                    dcc.Dropdown(
                        id="geospatial-metric",
                        options=[{"label": m.capitalize(), "value": m} for m in ["cases", "hospitalizations", "vaccination_rate"]],
                        value="cases",
                        clearable=False,
                        className="w-48 mb-4"
                    ),
                    dcc.Graph(id="geospatial-choropleth", style={"height": "500px"})
                ]
            ),
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("Regional Scatter Analysis", className="text-lg font-semibold mb-4"),
                    dcc.Graph(id="geospatial-scatter", style={"height": "400px"})
                ]
            ),
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("Cases by District with Population Density", className="text-lg font-semibold mb-4"),
                    dcc.Graph(id="geospatial-bubble", style={"height": "400px"})
                ]
            ),
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("Cases Density Heatmap", className="text-lg font-semibold mb-4"),
                    dcc.Graph(id="geospatial-density", style={"height": "400px"})
                ]
            )
        ]
    )
])
# 5. PREDICTIONS PAGE
predictions_layout = lambda pathname="/": html.Div([
    html.Div(
        className="container mx-auto px-4",
        children=[
            html.Div(
                className="mb-6",
                children=[
                    html.H1("Predictive Insights", className="text-2xl font-bold text-gray-800"),
                    html.P("Forecasted trends and relationships based on historical data", className="text-gray-600")
                ]
            ),
            create_filter_bar(),
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("Cases Forecast (Next 6 Months)", className="text-lg font-semibold mb-4"),
                    dcc.Graph(id="predictions-forecast-chart", style={"height": "400px"})
                ]
            ),
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("3D Relationship Analysis (Scatter)", className="text-lg font-semibold mb-4"),
                    dcc.Graph(id="predictions-3d-chart", style={"height": "500px"})
                ]
            ),
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("Parallel Coordinates by Region", className="text-lg font-semibold mb-4"),
                    dcc.Graph(id="predictions-parallel", style={"height": "400px"})
                ]
            ),
            html.Div(
                className="bg-white rounded-lg shadow-md p-4 border border-gray-200 mb-6",
                children=[
                    html.H3("3D Surface: Cases vs Vaccination & Access", className="text-lg font-semibold mb-4"),
                    dcc.Graph(id="predictions-surface", style={"height": "500px"})
                ]
            )
        ]
    )
])

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# Callback to render page content based on URL
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    navbar = create_navbar(pathname)

    if pathname == "/trends":
        return html.Div([navbar, trends_layout(pathname)])  
    elif pathname == "/comparisons":
        return html.Div([navbar, comparisons_layout(pathname)])  
    elif pathname == "/geospatial":
        return html.Div([navbar, geospatial_layout(pathname)])  
    elif pathname == "/predictions":
        return html.Div([navbar, predictions_layout(pathname)])  
    elif pathname == "/" or pathname is None:
        return html.Div([navbar, overview_layout(pathname)]) 
    else:
        return html.Div([navbar, overview_layout(pathname)])  

    
# Helper function to filter data based on user inputs
def filter_data(start_date, end_date, regions, conditions, age_groups, hospital_types):
    filtered_df = df.copy()
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df["date"] >= pd.to_datetime(start_date)) &
            (filtered_df["date"] <= pd.to_datetime(end_date))
        ]
    if regions:
        filtered_df = filtered_df[filtered_df["region"].isin(regions)]
    if conditions:
        filtered_df = filtered_df[filtered_df["condition"].isin(conditions)]
    if age_groups:
        filtered_df = filtered_df[filtered_df["age_group"].isin(age_groups)]
    if hospital_types:
        filtered_df = filtered_df[filtered_df["hospital_type"].isin(hospital_types)]
    return filtered_df

@app.callback(
    [
        Output("overview-trend-chart", "figure"),
        Output("overview-distribution-chart", "figure"),
        Output("overview-hosp-ratio-chart", "figure"),
        Output("overview-age-chart", "figure"),
        Output("overview-hospital-chart", "figure"),
        Output("overview-calendar-chart", "figure")
    ],
    [
        Input("apply-filters-button", "n_clicks"),
        State("date-range-filter", "start_date"),
        State("date-range-filter", "end_date"),
        State("region-filter", "value"),
        State("condition-filter", "value"),
        State("age-filter", "value"),
        State("hospital-filter", "value"),
        State("url", "pathname")
    ]
)
def update_overview_charts(n_clicks, start_date, end_date, regions, conditions, age_groups, hospital_types, pathname):
    if pathname not in ["/", None]:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()
    
    filtered_df = filter_data(start_date, end_date, regions, conditions, age_groups, hospital_types)
    
    # 1. Trend Chart (Cases Over Time by Condition)
    trend_fig = px.line(
        filtered_df.groupby(["date", "condition"]).sum(numeric_only=True).reset_index(),
        x="date", y="cases", color="condition",
        title="Cases Over Time by Condition"
    )
    
    # 2. Distribution Chart (Case Distribution by Region)
    dist_df = filtered_df.groupby("region").sum(numeric_only=True).reset_index()
    colors = ['#2ecc71', '#3498db', '#2980b9', '#27ae60', '#1abc9c'] 
    dist_fig = go.Figure(data=[go.Pie(
        labels=dist_df["region"],
        values=dist_df["cases"],
        hole=0.3,  
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="percent+label",  
        textposition="outside",  
        hoverinfo="label+value+percent",  
        pull=[0.05] * len(dist_df),  
    )])
    dist_fig.update_layout(
        title="Case Distribution by Region",
        paper_bgcolor="#ffffff", 
        plot_bgcolor="#ffffff",  
        font=dict(color="#1a1a1a"),  
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # 3. Hospitalization Ratio Chart (Hospitalizations vs Cases)
    ratio_fig = px.scatter(
        filtered_df.groupby("date").mean(numeric_only=True).reset_index(),
        x="cases", y="hospitalizations",
        title="Hospitalizations vs Cases"
    )
    
    # 4. Age Group Chart (Case Distribution by Age Group)
    age_fig = px.bar(
        filtered_df.groupby("age_group").sum(numeric_only=True).reset_index(),
        x="age_group", y="cases",
        title="Case Distribution by Age Group",
        color="age_group",
        text_auto=".2s"
    )
    age_fig.update_traces(textposition="auto")
    age_fig.update_layout(showlegend=False)
    
    # 5. Hospital Type Chart (Metrics by Hospital Type)
    hospital_df = filtered_df.groupby(["hospital_type", "date"]).mean(numeric_only=True).reset_index()

    hospital_monthly_df = hospital_df.copy()
    hospital_monthly_df['month_year'] = hospital_monthly_df['date'].dt.strftime('%Y-%m')
    hospital_monthly_df = hospital_monthly_df.groupby(['hospital_type', 'month_year']).mean(numeric_only=True).reset_index()

    hospital_fig = go.Figure()

    colors = ['rgba(0, 191, 255, 0.7)', 'rgba(64, 224, 208, 0.6)', 'rgba(128, 0, 128, 0.5)', 'rgba(255, 165, 0, 0.6)']
    hospital_types = hospital_monthly_df["hospital_type"].unique()

    for i, hospital_type in enumerate(hospital_types):
        subset = hospital_monthly_df[hospital_monthly_df["hospital_type"] == hospital_type]
        subset = subset.sort_values('month_year')
        hospital_fig.add_trace(go.Scatter(
            x=subset["month_year"],
            y=subset["satisfaction_score"],
            fill='tozeroy',
            mode='lines',
            line=dict(shape="spline", smoothing=1.3, width=2, color=colors[i % len(colors)]),
            name=hospital_type,
            fillcolor=colors[i % len(colors)],
            hovertemplate=f"{hospital_type}<br>Date: %{{x}}<br>Satisfaction: %{{y:.2f}}"
        ))

        hospital_fig.update_layout(
        title="Hospital Satisfaction Trends",
        title_font=dict(size=20, color='black'),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(color='black', family="Arial"),
        legend=dict(orientation="v", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='black', size=12), bgcolor='white'),
        xaxis=dict(title="", tickformat="%b %Y", tickangle=-45, tickcolor='black', tickfont=dict(size=10, color='black'), showgrid=False, zeroline=False, showline=False),
        yaxis=dict(title="", range=[0, 5], tickcolor='black', tickfont=dict(size=10, color='black'), showgrid=True, gridwidth=0.5, gridcolor='rgba(0, 0, 0, 0.1)', zeroline=False, showline=False),
        margin=dict(l=10, r=10, t=50, b=30),
        height=400
    )
    # 6. Calendar Chart (Seasonal Patterns of Cases)
    calendar_fig = px.density_heatmap(
        filtered_df.groupby(["year", "month"]).sum(numeric_only=True).reset_index(),
        x="month", y="year", z="cases",
        title="Seasonal Patterns of Cases",
        color_continuous_scale="Viridis",
        labels={"month": "Month", "year": "Year", "cases": "Total Cases"}
    )
    calendar_fig.update_layout(xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]))
    
    return trend_fig, dist_fig, ratio_fig, age_fig, hospital_fig, calendar_fig

# Trends Callback
@app.callback(
    [
        Output("trends-time-series-chart", "figure"),
        Output("trends-animated-chart", "figure"),    
        Output("trends-seasonal-chart", "figure"),    
        Output("trends-rolling-chart", "figure")      
        
    ],
    [
        Input("apply-filters-button", "n_clicks"),
        Input("trends-time-grouping", "value"),
        Input("trends-metrics-selector", "value"),
        Input("trends-x-axis", "value"),             
        Input("trends-y-axis", "value"),             
        Input("trends-seasonal-metric", "value"),     
        Input("trends-rolling-window", "value"),      
        State("date-range-filter", "start_date"),
        State("date-range-filter", "end_date"),
        State("region-filter", "value"),
        State("condition-filter", "value"),
        State("age-filter", "value"),
        State("hospital-filter", "value"),
        State("url", "pathname")
    ]
)

def update_trends_time_series(n_clicks, grouping, metrics, x_axis, y_axis, seasonal_metric, rolling_window, 
                            start_date, end_date, regions, conditions, age_groups, hospital_types, pathname):
    if pathname != "/trends":
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    
    filtered_df = filter_data(start_date, end_date, regions, conditions, age_groups, hospital_types)

    
    # 1. Time Series Chart (Multi-Metric Time Series Analysis)
    grouped_df = filtered_df.groupby(pd.Grouper(key="date", freq=grouping)).sum(numeric_only=True).reset_index()
    time_series_fig = make_subplots(specs=[[{"secondary_y": True}]])
    for metric in metrics:
        time_series_fig.add_trace(
            go.Scatter(x=grouped_df["date"], y=grouped_df[metric], name=metric.capitalize()),
            secondary_y=(metric in ["recovery_days", "treatment_cost"])
        )
    time_series_fig.update_layout(title="Multi-Metric Time Series")


    
    # 2. Animated Chart (Regional Metrics Evolution)
    animated_fig = px.scatter(
        filtered_df.groupby(["region", "year"]).mean(numeric_only=True).reset_index(),
        x=x_axis, y=y_axis, color="region", animation_frame="year",
        title=f"{y_axis.capitalize()} vs {x_axis.capitalize()} Over Time",
        size="cases", hover_data=["region"]
    )
    animated_fig.update_layout(transition={"duration": 500})


    
    # 3. Seasonal Chart (Seasonal Patterns)
    seasonal_df = filtered_df.groupby(["month"]).mean(numeric_only=True).reset_index()
    seasonal_fig = px.line(
        seasonal_df,
        x="month", y=seasonal_metric,
        title=f"Seasonal Pattern of {seasonal_metric.capitalize()}",
        labels={"month": "Month"}
    )
    seasonal_fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    )

    
    # 4. Rolling Chart (Rolling Average Trends)
    rolling_df = filtered_df.groupby("date").sum(numeric_only=True).reset_index()
    rolling_df["cases_rolling"] = rolling_df["cases"].rolling(window=rolling_window, min_periods=1).mean()
    rolling_fig = px.line(
        rolling_df,
        x="date", y="cases_rolling",
        title=f"Rolling Average Cases (Window: {rolling_window} Days)"
    )
    
    return time_series_fig, animated_fig, seasonal_fig, rolling_fig
# Comparisons Callback
@app.callback(
    [
        Output("comparisons-bar-chart", "figure"),
        Output("comparisons-box-chart", "figure"),
        Output("comparisons-heatmap", "figure"),
        Output("comparisons-treemap", "figure"),  
        Output("comparisons-sankey", "figure")    
    ],
    [
        Input("apply-filters-button", "n_clicks"),
        Input("comparisons-metric", "value"),
        State("date-range-filter", "start_date"),
        State("date-range-filter", "end_date"),
        State("region-filter", "value"),
        State("condition-filter", "value"),
        State("age-filter", "value"),
        State("hospital-filter", "value"),
        State("url", "pathname")
    ]
)
def update_comparisons_charts(n_clicks, metric, start_date, end_date, regions, conditions, age_groups, hospital_types, pathname):
    if pathname != "/comparisons":
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()
    
    filtered_df = filter_data(start_date, end_date, regions, conditions, age_groups, hospital_types)
    
    # 1. Bar Chart
    bar_fig = px.bar(
     filtered_df.groupby("region").sum(numeric_only=True).reset_index(),
     y="region", 
     x=metric,    
     title=f"{metric.capitalize()} by Region",
     color="region",  
     text_auto=".2s",  
     orientation='h'   
 )
     

    bar_fig.update_traces(textposition="inside",width=1.2)
     

    bar_fig.update_layout(
     showlegend=False,  
     xaxis_title=metric.capitalize(), 
     yaxis_title="Region",  
     xaxis=dict(showgrid=False),
     yaxis=dict(showgrid=False),  
     plot_bgcolor='rgba(0,0,0,0)',
     paper_bgcolor='rgba(0,0,0,0)',  
     font=dict(size=12) , 
     bargap=0.1
 )
    
    # 2. Box Chart
    box_fig = px.box(
        filtered_df,
        x="condition", y=metric, title=f"Distribution of {metric.capitalize()} by Condition",
        color="condition", points="outliers"
    )
    box_fig.update_layout(showlegend=False)
    
    # 3. Heatmap
    numeric_cols = ["cases", "hospitalizations", "icu_admissions", "recovery_days", "treatment_cost", "vaccination_rate", "satisfaction_score"]
    corr_df = filtered_df[numeric_cols].corr()
    heatmap_fig = px.imshow(
        corr_df, text_auto=".2f", title="Correlation Heatmap of Key Metrics",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1
    )
    heatmap_fig.update_layout(height=500)
    
    # 4. Treemap
    treemap_df = filtered_df.groupby(["region", "condition"]).sum(numeric_only=True).reset_index()
    treemap_fig = px.treemap(
        treemap_df,
        path=["region", "condition"], values="cases",
        title="Cases by Region and Condition",
        color="cases", color_continuous_scale="Blues"
    )
    
    # 5. Sankey Diagram
    sankey_df = filtered_df.groupby("region").sum(numeric_only=True).reset_index()
    labels = list(sankey_df["region"]) + ["Cases", "Hospitalizations", "ICU Admissions"]
    source = []
    target = []
    value = []
    for i, region in enumerate(sankey_df["region"]):
        source.extend([i, len(sankey_df) + 0, len(sankey_df) + 1])  
        target.extend([len(sankey_df) + 0, len(sankey_df) + 1, len(sankey_df) + 2])
        value.extend([sankey_df.loc[i, "cases"], sankey_df.loc[i, "hospitalizations"], sankey_df.loc[i, "icu_admissions"]])
    sankey_fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color="blue"),
        link=dict(source=source, target=target, value=value)
    ))
    sankey_fig.update_layout(title="Patient Flow: Cases to Hospitalizations to ICU")
    
    return bar_fig, box_fig, heatmap_fig, treemap_fig, sankey_fig


# Geospatial Callback
@app.callback(
    [
        Output("geospatial-choropleth", "figure"),
        Output("geospatial-scatter", "figure"),
        Output("geospatial-bubble", "figure"),    
        Output("geospatial-density", "figure")    
    ],
    [
        Input("apply-filters-button", "n_clicks"),
        Input("geospatial-metric", "value"),
        State("date-range-filter", "start_date"),
        State("date-range-filter", "end_date"),
        State("region-filter", "value"),
        State("condition-filter", "value"),
        State("age-filter", "value"),
        State("hospital-filter", "value"),
        State("url", "pathname")
    ]
)
def update_geospatial_charts(n_clicks, metric, start_date, end_date, regions, conditions, age_groups, hospital_types, pathname):
    if pathname != "/geospatial":
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    
    filtered_df = filter_data(start_date, end_date, regions, conditions, age_groups, hospital_types)
    
    # 1. Choropleth Map
    choropleth_df = filtered_df.groupby("district").agg({
        metric: "sum", "latitude": "mean", "longitude": "mean"
    }).reset_index()
    choropleth_fig = px.scatter_mapbox(
        choropleth_df,
        lat="latitude", lon="longitude", size=metric, color=metric,
        hover_name="district", title=f"{metric.capitalize()} by District",
        mapbox_style="carto-positron", zoom=3
    )
    
    # 2. Scatter Geo Plot
    scatter_df = filtered_df.groupby("region").agg({
        metric: "mean", "latitude": "mean", "longitude": "mean", "cases": "sum"
    }).reset_index()
    scatter_fig = px.scatter_geo(
        scatter_df,
        lat="latitude", lon="longitude", size="cases", color=metric,
        hover_name="region", title=f"{metric.capitalize()} vs Cases by Region",
        projection="natural earth"
    )
    
    # 3. Bubble Map
    bubble_df = filtered_df.groupby("district").agg({
        "cases": "sum", "latitude": "mean", "longitude": "mean", "population_density": "mean"
    }).reset_index()
    bubble_fig = px.scatter_mapbox(
        bubble_df,
        lat="latitude", lon="longitude", size="cases", color="population_density",
        hover_name="district", title="Cases by District with Population Density",
        mapbox_style="open-street-map", zoom=3, color_continuous_scale="Viridis"
    )
    
    # 4. Density Mapbox
    density_fig = px.density_mapbox(
        filtered_df,
        lat="latitude", lon="longitude", z="cases",
        radius=20, center=dict(lat=37.5, lon=-97.5), zoom=3,
        mapbox_style="carto-positron", title="Cases Density Heatmap"
    )
    
    return choropleth_fig, scatter_fig, bubble_fig, density_fig

# Predictions Callback

@app.callback(
    [
        Output("predictions-forecast-chart", "figure"),
        Output("predictions-3d-chart", "figure"),
        Output("predictions-parallel", "figure"),  
        Output("predictions-surface", "figure")    
    ],
    [
        Input("apply-filters-button", "n_clicks"),
        State("date-range-filter", "start_date"),
        State("date-range-filter", "end_date"),
        State("region-filter", "value"),
        State("condition-filter", "value"),
        State("age-filter", "value"),
        State("hospital-filter", "value"),
        State("url", "pathname")
    ]
)
def update_predictions_charts(n_clicks, start_date, end_date, regions, conditions, age_groups, hospital_types, pathname):
    if pathname != "/predictions":
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    
    filtered_df = filter_data(start_date, end_date, regions, conditions, age_groups, hospital_types)
    
    # 1. Forecast Chart
    grouped_df = filtered_df.groupby("date").sum(numeric_only=True).reset_index()
    last_date = grouped_df["date"].max()
    future_dates = pd.date_range(start=last_date, periods=180, freq="D")[1:]
    last_cases = grouped_df["cases"].iloc[-1]
    trend = (grouped_df["cases"].iloc[-1] - grouped_df["cases"].iloc[-30]) / 30
    future_cases = [last_cases + trend * i for i in range(1, 180)]
    forecast_df = pd.DataFrame({"date": future_dates, "cases": future_cases})
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=grouped_df["date"], y=grouped_df["cases"], name="Historical", mode="lines"))
    forecast_fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["cases"], name="Forecast", mode="lines", line=dict(dash="dash")))
    forecast_fig.update_layout(title="Cases Forecast (Next 6 Months)")
    
    # 2. 3D Scatter Chart
    scatter_df = filtered_df.groupby("region").mean(numeric_only=True).reset_index()
    scatter_fig = go.Figure(data=[go.Scatter3d(
        x=scatter_df["vaccination_rate"], y=scatter_df["healthcare_access_index"], z=scatter_df["cases"],
        mode="markers", marker=dict(size=5, color=scatter_df["cases"], colorscale="Viridis", showscale=True)
    )])
    scatter_fig.update_layout(
        title="3D Analysis: Cases vs Vaccination Rate vs Healthcare Access",
        scene=dict(xaxis_title="Vaccination Rate", yaxis_title="Healthcare Access Index", zaxis_title="Cases")
    )
    
    # 3. Parallel Coordinates
    parallel_df = filtered_df.groupby("region").mean(numeric_only=True).reset_index()
    parallel_fig = px.parallel_coordinates(
        parallel_df,
        dimensions=["cases", "hospitalizations", "icu_admissions", "vaccination_rate", "treatment_cost"],
        color="cases", labels={col: col.capitalize() for col in parallel_df.columns},
        title="Parallel Coordinates of Key Metrics by Region",
        color_continuous_scale=px.colors.diverging.Tealrose
    )
    
    # 4. 3D Surface
    # Create a grid for interpolation
    x = scatter_df["vaccination_rate"]
    y = scatter_df["healthcare_access_index"]
    z = scatter_df["cases"]
    xi = np.linspace(x.min(), x.max(), 20)
    yi = np.linspace(y.min(), y.max(), 20)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method="cubic")
    surface_fig = go.Figure(data=[go.Surface(
        x=xi, y=yi, z=zi,
        colorscale="Viridis", showscale=True
    )])
    surface_fig.update_layout(
        title="3D Surface: Cases vs Vaccination Rate vs Healthcare Access",
        scene=dict(xaxis_title="Vaccination Rate", yaxis_title="Healthcare Access Index", zaxis_title="Cases")
    )
    
    return forecast_fig, scatter_fig, parallel_fig, surface_fig




#  custom CSS

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Health Analytics Dashboard</title>
        {%favicon%}
        {%css%}
        <!-- Add Google Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            /* Apply Poppins to all elements */
            * {
                font-family: 'Poppins', sans-serif;
                
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #1a1a1a;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
            
            /* Animations */
            .animate-pulse {
                animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            }
            
            @keyframes pulse {
                0%, 100% {
                    opacity: 1;
                }
                50% {
                    opacity: .7;
                }
            }
            
            /* Card hover effects */
            .card-hover {
                transition: transform 0.3s ease-in-out;
            }
            
            .card-hover:hover {
                transform: translateY(-5px);
            }
            
            /* Font weight utilities */
            .font-light {
                font-weight: 300;
            }
            
            .font-regular {
                font-weight: 400;
            }
            
            .font-medium {
                font-weight: 500;
            }
            
            .font-semibold {
                font-weight: 600;
            }
            
            .font-bold {
                font-weight: 700;
            }
            
            /* Additional font styling for specific elements */
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Poppins', sans-serif;
                font-weight: 600;
            }
            
            .metric-value {
                font-family: 'Poppins', sans-serif;
                font-weight: 500;
            }
            
            .metric-label {
                font-family: 'Poppins', sans-serif;
                font-weight: 400;
            }

            /* Add this to your existing styles in app.index_string */
            .Select-control {
                background-color: #1f2937 !important;
                border-color: #374151 !important;
                border-radius: 0.375rem !important;
            }

            .Select-menu-outer {
                background-color: #1f2937 !important;
                border: 1px solid #374151 !important;
                border-radius: 0.375rem !important;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
            }

            .Select-option {
                background-color: #1f2937 !important;
                color: white !important;
                padding: 0.5rem 1rem !important;
            }

            .Select-option:hover {
                background-color: #2d3748 !important;
            }

            .Select-value-label {
                color: white !important;
            }

            .Select-placeholder {
                color: #9ca3af !important;
            }

            .Select-arrow-zone {
                color: #9ca3af !important;
            }

            .Select.is-focused > .Select-control {
                border-color: #3b82f6 !important;
                box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)