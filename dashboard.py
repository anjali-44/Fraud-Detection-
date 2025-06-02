from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv("synthetic_data.csv")

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Admin Dashboard - Fraud Detection"
server = app.server  # for deployment

# Preprocessing
fraud_counts = df['label'].value_counts()
fraud_data = pd.DataFrame({
    "Type": ["Non-Fraudulent", "Fraudulent"],
    "Count": fraud_counts.values
})

device_distribution = df['device'].value_counts()
device_data = pd.DataFrame({
    "Device": device_distribution.index,
    "Count": device_distribution.values
})

# Graphs
fraud_trend_chart = px.bar(
    fraud_data, x="Type", y="Count", title="Fraud vs Non-Fraud Transactions",
    color="Type", text="Count",
    color_discrete_map={"Non-Fraudulent": "#00cc96", "Fraudulent": "#EF553B"}
).update_layout(title_x=0.5)

amount_histogram = px.histogram(
    df, x="amount", color="label", nbins=30,
    title="Transaction Amount Distribution",
    color_discrete_map={0: "#636EFA", 1: "#AB63FA"}
).update_layout(title_x=0.5)

device_pie_chart = px.pie(
    device_data, names="Device", values="Count", title="Device Usage Distribution"
).update_traces(textinfo="percent+label").update_layout(title_x=0.5)

# Layout
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col([
            dbc.NavbarSimple(
                brand="🛡️ Fraud Detection Admin",
                color="primary", dark=True, fluid=True, className="mb-4"
            )
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Transactions", className="card-title"),
                    html.H2(f"{len(df):,}", className="card-text")
                ])
            ], color="dark", inverse=True)
        ]),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Fraudulent Transactions", className="card-title"),
                    html.H2(f"{fraud_counts.get(1, 0):,}", className="card-text")
                ])
            ], color="danger", inverse=True)
        ]),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Non-Fraudulent Transactions", className="card-title"),
                    html.H2(f"{fraud_counts.get(0, 0):,}", className="card-text")
                ])
            ], color="success", inverse=True)
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Fraud Trends"),
                dbc.CardBody([
                    dcc.Graph(figure=fraud_trend_chart)
                ])
            ])
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Transaction Amount Distribution"),
                dbc.CardBody([
                    dcc.Graph(figure=amount_histogram)
                ])
            ])
        ], md=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Device Usage Breakdown"),
                dbc.CardBody([
                    dcc.Graph(figure=device_pie_chart)
                ])
            ])
        ], md=6)
    ]),

    dbc.Row([
        dbc.Col(html.Footer(
            "© 2025 Fraud Detection System. All rights reserved.",
            className="text-center mt-4 text-light small"
        ))
    ])
])


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
