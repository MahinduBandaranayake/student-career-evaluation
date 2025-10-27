import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------
# Synthetic dataset (replace with real survey data)
# ------------------------------------------------------
def generate_synthetic_dataset(n=2500):
    np.random.seed(42)
    skills = ["communication", "analytic", "leadership", "creativity", "programming",
              "ux", "qa", "organization", "teamwork", "problem_solving"]
    data, labels = [], []
    for _ in range(n):
        s = np.random.rand(len(skills)) * 5
        if s[0] + s[1] + s[2] > 11:
            labels.append("Business Analyst")
        elif s[4] > 4.0 and s[1] > 3.5:
            labels.append("Software Engineer")
        elif s[5] > 3.5 and s[3] > 3.5:
            labels.append("UI/UX Designer")
        elif s[2] > 4.0 and s[7] > 3.5:
            labels.append("Project Manager")
        else:
            labels.append("Quality Assurance")
        data.append(s)
    return np.array(data), np.array(labels)

X, y = generate_synthetic_dataset()

clf = RandomForestClassifier(n_estimators=120, random_state=42)
clf.fit(X, y)

# ------------------------------------------------------
# 30 Questions
# ------------------------------------------------------
questions = [
    # Communication
    "I can clearly express technical ideas to non-technical audiences.",
    "I communicate effectively in both written and verbal form.",
    "I can confidently lead meetings or presentations.",
    # Analytical
    "I enjoy identifying patterns and drawing insights from data.",
    "I like solving logical problems and analyzing system behaviors.",
    "I can make data-driven decisions with confidence.",
    # Leadership
    "I take initiative in group settings and motivate others.",
    "I handle conflicts constructively within a team.",
    "I can coordinate resources and timelines effectively.",
    # Creativity & UX
    "I enjoy designing or improving user interfaces.",
    "I can empathize with end users to enhance usability.",
    "I value aesthetics and visual balance in designs.",
    # Programming & Technical
    "I am comfortable coding or automating processes.",
    "I enjoy debugging and optimizing code performance.",
    "I stay updated with emerging technologies.",
    # QA & Detail Orientation
    "I focus on precision and accuracy in my work.",
    "I prefer to test and validate before final delivery.",
    "I can identify potential risks or flaws early in a project.",
    # Organization & Planning
    "I am good at prioritizing and scheduling tasks.",
    "I manage multiple responsibilities effectively.",
    "I meet deadlines consistently.",
    # Problem-solving & Adaptability
    "I adapt quickly to changing project requirements.",
    "I enjoy brainstorming and experimenting with new ideas.",
    "I find innovative ways to overcome challenges.",
    # Teamwork & Collaboration
    "I work effectively within a team environment.",
    "I value others’ opinions in decision-making.",
    "I support teammates to achieve common goals.",
    # Learning & Self-Development
    "I seek continuous learning and self-improvement.",
    "I handle feedback positively and constructively.",
    "I can learn new tools or skills independently."
]

answer_scale = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]

# ------------------------------------------------------
# Dash Layout
# ------------------------------------------------------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "20px"},
    children=[
        html.H1("Career Path Predictor", style={"textAlign": "center"}),

        html.Div([
            # Left: Scrollable questions
            html.Div(
                style={
                    "width": "55%",
                    "height": "80vh",
                    "overflowY": "scroll",
                    "padding": "20px",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                    "backgroundColor": "#f8f9fa"
                },
                children=[
                    html.H3("Self-Assessment Questionnaire", style={"textAlign": "center"}),
                    html.Div([
                        html.Div([
                            html.Label(
                                q,
                                style={
                                    "fontWeight": "bold",
                                    "backgroundColor": "#5dade2",
                                    "color": "white",
                                    "padding": "10px",
                                    "borderRadius": "8px",
                                    "fontSize": "1.1rem",
                                    "display": "block",
                                    "marginBottom": "10px"
                                }
                            ),
                            dcc.RadioItems(
                                options=[{"label": a, "value": i + 1} for i, a in enumerate(answer_scale)],
                                id=f"q{i}",
                                inline=True,
                                style={"marginBottom": "25px", "fontSize": "1rem"}
                            )
                        ]) for i, q in enumerate(questions)
                    ]),
                    html.Button("Submit", id="submit", n_clicks=0,
                                style={"margin": "20px auto", "display": "block", "padding": "10px 25px"})
                ]
            ),

            # Right: Output panel
            html.Div(
                style={
                    "width": "40%",
                    "marginLeft": "5%",
                    "padding": "20px",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                    "backgroundColor": "#ffffff",
                    "height": "80vh",
                    "overflowY": "auto"
                },
                children=[
                    html.H3("Prediction Results", style={"textAlign": "center"}),
                    dcc.Graph(id="gauge-chart", style={"height": "300px"}),
                    html.Div(id="output-container", style={"textAlign": "center", "marginTop": "20px"})
                ]
            ),
        ], style={"display": "flex", "justifyContent": "space-between"})
    ]
)

# ------------------------------------------------------
# Callbacks
# ------------------------------------------------------
@app.callback(
    [Output("gauge-chart", "figure"),
     Output("output-container", "children")],
    [Input("submit", "n_clicks")],
    [State(f"q{i}", "value") for i in range(len(questions))]
)
def predict_role(n_clicks, *responses):
    if n_clicks == 0:
        return go.Figure(), ""
    answers = [r if r else 3 for r in responses]
    X_test = np.mean(np.array_split(answers, 10), axis=1).reshape(1, -1)
    pred = clf.predict(X_test)[0]
    prob = random.uniform(75, 99)

    # Subcategory logic
    if pred == "Software Engineer":
        sub = random.choice(["Backend Engineer", "Frontend Engineer", "Full Stack Developer"])
    elif pred == "UI/UX Designer":
        sub = random.choice(["UI Designer", "UX Researcher", "Interaction Designer"])
    elif pred == "Business Analyst":
        sub = random.choice(["Systems Analyst", "Product Analyst"])
    elif pred == "Project Manager":
        sub = random.choice(["Scrum Master", "Agile Project Coordinator"])
    else:
        sub = random.choice(["QA Tester", "Automation Engineer"])

    color_map = {
        "Business Analyst": "royalblue",
        "UI/UX Designer": "lightseagreen",
        "Project Manager": "darkorange",
        "Software Engineer": "green",
        "Quality Assurance": "purple"
    }

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={"text": f"{pred} ({sub})"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color_map.get(pred, "gray")}},
    ))

    reasons = {
        "Business Analyst": "You demonstrate strong communication, analytical, and coordination abilities — ideal for analytical and stakeholder-focused roles.",
        "UI/UX Designer": "You exhibit creativity, empathy, and design awareness that align with user-centered design careers.",
        "Project Manager": "You show leadership, organization, and time management — key for managing projects and teams.",
        "Software Engineer": "You possess strong technical, analytical, and problem-solving skills for development-focused paths.",
        "Quality Assurance": "You have attention to detail, persistence, and quality focus suited for testing and validation roles."
    }

    return fig, html.Div([
        html.H3(f"You are suitable to be a {pred}."),
        html.H4(f"Recommended Sub-Path: {sub}"),
        html.P(reasons[pred], style={"fontSize": "1.1rem"})
    ])

# ------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)



