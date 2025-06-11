# For computations
import numpy as np
import pandas as pd
from scipy import stats

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

def compare_values_by_label(sensor, df):
    """
    Create an improved distribution comparison chart with multiple visualizations
    """

    # Extract data for each category
    fire_data = df[df['label'] == "Fire"][sensor].dropna().values
    non_fire_data = df[df['label'] == "Non-Fire"][sensor].dropna().values
    potential_fire_data = df[df['label'] == "Potential Fire"][sensor].dropna().values

    # Define colors for consistency
    colors = {
        'Fire': '#FF4B4B',           # Red
        'Non-Fire': '#4B8BFF',       # Blue
        'Potential Fire': '#FFB84B'  # Orange
    }

    # Create subplots: histogram + box plot
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f'{sensor} Distribution by Fire Classification', 'Box Plot Comparison']
    )

    # Calculate optimal bin size using Freedman-Diaconis rule
    all_data = np.concatenate([fire_data, non_fire_data, potential_fire_data])
    q75, q25 = np.percentile(all_data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(all_data) ** (1/3))

    # Add histograms with KDE curves
    datasets = [
        ('Fire', fire_data),
        ('Non-Fire', non_fire_data),
        ('Potential Fire', potential_fire_data)
    ]

    for name, data in datasets:
        if len(data) > 0:
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=name,
                    opacity=0.7,
                    nbinsx=max(10, int((data.max() - data.min()) / bin_width)),
                    marker_color=colors[name],
                    legendgroup=name,
                    histnorm='probability density'
                ),
                row=1, col=1
            )

            # Add KDE curve
            if len(data) > 1:
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                kde_values = kde(x_range)

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=kde_values,
                        mode='lines',
                        name=f'{name} KDE',
                        line=dict(color=colors[name], width=2),
                        legendgroup=name,
                        showlegend=False
                    ),
                    row=1, col=1
                )

            # Add box plot
            fig.add_trace(
                go.Box(
                    x=data,
                    name=name,
                    marker_color=colors[name],
                    legendgroup=name,
                    showlegend=False,
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=2, col=1
            )

    # Calculate statistics for annotation
    stats_text = []
    for name, data in datasets:
        if len(data) > 0:
            stats_text.append(
                f"<b>{name}</b><br>"
                f"Mean: {np.mean(data):.2f}<br>"
                f"Std: {np.std(data):.2f}<br>"
                f"Count: {len(data)}"
            )

    # Update layout
    fig.update_layout(
        title={
            'text': f'<b>Distribution Analysis: {sensor} by Fire Classification</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },

        # Histogram subplot
        xaxis=dict(title=f'{sensor} Value'),
        yaxis=dict(title='Probability Density'),

        # Box plot subplot
        xaxis2=dict(title=f'{sensor} Value'),
        yaxis2=dict(title='Categories', showticklabels=False),

        # General layout
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),

        # Add statistics annotation
        annotations=[
            dict(
                text="<br>".join(stats_text),
                showarrow=False,
                xref="paper", yref="paper",
                x=1.15, y=0.5,
                xanchor="left",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        ],

        # Professional styling
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12)
    )

    # Update axes styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig

def create_class_imbalance_bar_chart(df):
    """
    Create a bar chart showing class imbalance
    """
    class_counts = df['label'].value_counts()
    
    colors = ['#FF4B4B' if label == 'Fire' 
              else '#4B8BFF' if label == 'Non-Fire' 
              else '#FFB84B' for label in class_counts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=class_counts.index,
            y=class_counts.values,
            marker_color=colors,
            text=class_counts.values,
            textposition='auto',
            name='Class Distribution'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '<b>Class Distribution: Fire Event Categories</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='Fire Event Category',
        yaxis_title='Number of Samples',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=False,
        height=400
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_class_percentage_donut_chart(df):
    """
    Create a donut chart showing class percentages
    """
    class_counts = df['label'].value_counts()
    class_percentages = (class_counts / class_counts.sum() * 100).round(1)
    
    colors = ['#FF4B4B' if label == 'Fire' 
              else '#4B8BFF' if label == 'Non-Fire' 
              else '#FFB84B' for label in class_counts.index]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=class_counts.index,
            values=class_counts.values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '<b>Class Distribution Percentages</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        height=400,
        annotations=[dict(text='Fire<br>Categories', x=0.5, y=0.5, font_size=14, showarrow=False)]
    )
    
    return fig

def create_parallel_coordinates_plot(df, title_suffix=""):
    """
    Create a parallel coordinates plot for multidimensional data visualization
    """
    # Define colors for each class and create numerical mapping
    color_map = {'Fire': 2, 'Non-Fire': 0, 'Potential Fire': 1}
    
    # Create numerical labels for color mapping
    df_viz = df.copy()
    df_viz['color_num'] = df_viz['label'].map(color_map)
    
    # Create the parallel coordinates plot
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df_viz['color_num'],
                colorscale=[[0, '#4B8BFF'], [0.5, '#FFB84B'], [1, '#FF4B4B']],
                showscale=True,
                colorbar=dict(
                    title="Fire Categories",
                    tickvals=[0, 1, 2],
                    ticktext=['Non-Fire', 'Potential Fire', 'Fire'],
                    len=0.8,
                    x=1.02
                )
            ),
            dimensions=[
                dict(
                    range=[df['temperature'].min(), df['temperature'].max()],
                    label='Temperature (°C)', 
                    values=df['temperature']
                ),
                dict(
                    range=[df['air_quality'].min(), df['air_quality'].max()],
                    label='Air Quality', 
                    values=df['air_quality']
                ),
                dict(
                    range=[df['carbon_monoxide'].min(), df['carbon_monoxide'].max()],
                    label='Carbon Monoxide (ppm)', 
                    values=df['carbon_monoxide']
                ),
                dict(
                    range=[df['gas_and_smoke'].min(), df['gas_and_smoke'].max()],
                    label='Gas & Smoke', 
                    values=df['gas_and_smoke']
                )
            ]
        )
    )
    
    fig.update_layout(
        title={
            'text': f'<b>Parallel Coordinates Plot: Multi-dimensional Data Visualization {title_suffix}</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        height=500,
        margin=dict(r=120)  # Add right margin for colorbar
    )
    
    return fig

def create_3d_scatter_plot(df, title_suffix=""):
    """
    Create a 3D scatter plot for three key sensor measurements
    """
    color_map = {'Fire': '#FF4B4B', 'Non-Fire': '#4B8BFF', 'Potential Fire': '#FFB84B'}
    
    fig = go.Figure()
    
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        fig.add_trace(
            go.Scatter3d(
                x=subset['temperature'],
                y=subset['carbon_monoxide'],
                z=subset['gas_and_smoke'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=color_map[label],
                    opacity=0.7,
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                name=label,
                hovertemplate=
                '<b>%{fullData.name}</b><br>' +
                'Temperature: %{x}°C<br>' +
                'Carbon Monoxide: %{y} ppm<br>' +
                'Gas & Smoke: %{z}<br>' +
                '<extra></extra>'
            )
        )
    
    fig.update_layout(
        title={
            'text': f'<b>3D Visualization: Sensor Measurements by Fire Categories {title_suffix}</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title='Temperature (°C)',
            yaxis_title='Carbon Monoxide (ppm)',
            zaxis_title='Gas & Smoke',
            bgcolor='white'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_data_quality_summary_table(quality_report):
    """
    Create a visual summary table for data quality assessment
    """
    # Prepare data for the table
    missing_values = quality_report['missing_values']
    data_types = quality_report['data_types']
    
    table_data = []
    for feature in missing_values.index:
        table_data.append([
            feature,
            str(data_types[feature]),
            missing_values[feature],
            "✓" if missing_values[feature] == 0 else "✗"
        ])
    
    # Add summary row
    table_data.append([
        "<b>SUMMARY</b>",
        f"<b>Shape: {quality_report['shape']}</b>",
        f"<b>Duplicates: {quality_report['duplicates']}</b>",
        "<b>✓ Data Ready</b>" if quality_report['duplicates'] == 0 and missing_values.sum() == 0 else "<b>⚠ Issues Found</b>"
    ])
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Feature</b>', '<b>Data Type</b>', '<b>Missing Values</b>', '<b>Status</b>'],
            fill_color='#800000',
            font_color='white',
            align='center',
            font_size=14
        ),
        cells=dict(
            values=list(zip(*table_data)),
            fill_color=[['white']*len(table_data[:-1]) + ['#f0f0f0']] * 4,
            align='center',
            font_size=12,
            height=30
        )
    )])
    
    fig.update_layout(
        title={
            'text': '<b>Data Quality Assessment Summary</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        height=300
    )
    
    return fig