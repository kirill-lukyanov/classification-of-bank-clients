import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import numpy as np
from scipy.stats import gaussian_kde

def distribution_vis(df: pd.DataFrame, x_features, y=None, is_category=False, title=None, width=1600, aspect_ratio=0.3):
    """
    Визуализация распределений признаков с использованием Plotly с гистограммами, KDE и boxplot

    Параметры:
    df - DataFrame с данными
    x_features - список признаков для визуализации
    y - целевая переменная (опционально)
    is_category - флаг категориальных признаков
    title - общий заголовок
    figsize - размер фигуры (ширина, высота) в пикселях
    height_per_plot - высота одного подграфика в пикселях
    aspect_ratio - соотношение сторон подграфика (width/height)
    """
    x_features = x_features.copy()
    if y is not None and y in x_features:
        x_features.remove(y)

    n_features = len(x_features)
    n_cols = min(n_features, 3)
    n_rows = math.ceil(n_features / n_cols)

    fig_width = width
    fig_height = n_rows * width * aspect_ratio

    # Создаем субплоты с дополнительным рядом для boxplot
    fig = make_subplots(
        rows=n_rows if is_category else n_rows*2,  # Удваиваем ряды (верхний - boxplot, нижний - гистограмма)
        cols=n_cols,
        subplot_titles=[feature for feature in x_features] if is_category else None,
        horizontal_spacing=0.1,
        vertical_spacing=0.04 if is_category else 0.075,
        row_heights=None if is_category else[0.2, 0.8]*n_rows  # 20% высоты на boxplot, 80% на гистограмму
    )

    for i, feature in enumerate(x_features):
        row_boxplot = (i // n_cols)+1 if is_category else (i // n_cols)*2 + 1  # Нечетные ряды (1, 3, 5...) для boxplot
        row_hist = row_boxplot + 1         # Четные ряды (2, 4, 6...) для гистограммы
        col = (i % n_cols) + 1

        if y is None:
            if is_category:
                # Для категориальных признаков - только barplot (без boxplot)
                counts = df[feature].value_counts().sort_index()
                fig.add_trace(
                    go.Bar(
                        x=counts.index,
                        y=counts.values,
                        name=feature,
                        showlegend=False
                    ),
                    row=row_boxplot, col=col
                )
                fig.update_yaxes(title_text="Количество", row=row_hist, col=col)
            else:
                # Для числовых признаков - boxplot + гистограмма с KDE
                x_data = df[feature].dropna()
                
                # Boxplot (верхний график)
                fig.add_trace(
                    go.Box(
                        x=x_data,
                        name=feature,
                        showlegend=False,
                        marker_color='#2ca02c',
                        line_color='#2ca02c',
                        # boxpoints=False  # Не показывать точки
                    ),
                    row=row_boxplot, col=col
                )
                
                # Гистограмма (нижний график)
                bin_width = (x_data.max() - x_data.min()) / 30
                fig.add_trace(
                    go.Histogram(
                        x=x_data,
                        name=feature,
                        xbins=dict(size=bin_width),
                        showlegend=False,
                        marker_color='#1f77b4',
                        opacity=0.7
                    ),
                    row=row_hist, col=col
                )
                
                # KDE
                kde = gaussian_kde(x_data)
                x_kde = np.linspace(x_data.min(), x_data.max(), 500)
                y_kde_density = kde(x_kde)
                y_kde_counts = (y_kde_density - np.min(y_kde_density)) / (np.max(y_kde_density) - np.min(y_kde_density)) * max(np.histogram(x_data, bins=30)[0])
                
                fig.add_trace(
                    go.Scatter(
                        x=x_kde,
                        y=y_kde_counts,
                        mode='lines',
                        line=dict(color='red', width=2),
                        hoverinfo='text',
                        text=[f'{d:.4f}' for d in y_kde_density],
                        name='KDE',
                        hovertemplate='<b>KDE</b><br>' + 'x: %{x:.2f}<br>' + 'Density: %{text}<extra></extra>',
                        showlegend=False
                    ),
                    row=row_hist, col=col
                )
                
                fig.update_yaxes(title_text="Плотность", row=row_hist, col=col)
                
                # Убираем оси X для boxplot (чтобы не дублировались)
                fig.update_xaxes(showticklabels=False, row=row_boxplot, col=col)
                fig.update_xaxes(title_text=feature, row=row_hist, col=col)
        else:
            if is_category:
                # Для категориальных признаков с целевой переменной
                median_values = df.groupby(feature)[y].median().sort_index()
                fig.add_trace(
                    go.Bar(
                        x=median_values.index,
                        y=median_values.values,
                        name=feature,
                        showlegend=False
                    ),
                    row=row_hist, col=col
                )
                fig.update_yaxes(title_text=f"Медиана {y}", row=row_hist, col=col)
            else:
                # Для числовых признаков с целевой переменной - scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=df[feature],
                        y=df[y],
                        mode='markers',
                        marker=dict(size=4, opacity=0.5),
                        name=feature,
                        showlegend=False
                    ),
                    row=row_hist, col=col
                )
                fig.update_yaxes(title_text=y, row=row_hist, col=col)

            fig.update_xaxes(title_text=feature, row=row_hist, col=col)

    # Обновление общего вида
    fig.update_layout(
        width=fig_width,
        height=fig_height,  # Увеличиваем высоту для boxplot
        title_text=title if title else "",
        title_x=0.5,
        margin=dict(l=50, r=50, b=50, t=50 if not title else 80),
        bargap=0.1,
        plot_bgcolor='white'
    )
    
    # Настройка отображения осей и сетки
    fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    fig.show()
    
def dependency_vis(X: pd.DataFrame, y: pd.Series, is_category=False, width=1600, title=None, aspect_ratio=0.25):
    cols = X.columns
    n_cols = 3
    if is_category:
        n_rows = math.ceil(len(cols) / 3)

        fig_width = width
        fig_height = n_rows * width * aspect_ratio
        
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols, vertical_spacing=0.08)
        for i, col in enumerate(cols):
            n_row = i // 3 + 1
            n_col = i % 3 + 1   
            
            cross_tab = pd.crosstab(index=X[col].astype(str), columns=y.astype('object'), normalize='index').sort_values(by=list(y.unique()))
            bar_traces = px.bar(cross_tab, orientation='h')
            if i > 0:
                bar_traces.update_traces(showlegend=False)
            if n_col == 1:
                fig.update_yaxes(title_text="Категория", row=n_row, col=n_col)
            fig.update_xaxes(title_text='Соотношение', row=n_row, col=n_col)
            fig.update_yaxes(tickvals=cross_tab.index, type='category', row=n_row, col=n_col)
            fig.add_traces(bar_traces.data, rows=n_row, cols=n_col)
    else:
        pairs = [(cols[n], cols[m]) for n in range(len(cols)) for m in range(n+1, len(cols))]
        n_rows = math.ceil(len(pairs) / 3)

        fig_width = width
        fig_height = n_rows * width * aspect_ratio
     
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=['-'.join(pair) for pair in pairs], vertical_spacing=0.06)
        for i, pair in enumerate(pairs):
            n_row = i // 3 + 1
            n_col = i % 3 + 1
            scatter_traces = px.scatter(x=X[pair[0]], y=X[pair[1]], color=y.astype('object'), opacity=0.5)
            if i > 0:
                scatter_traces.update_traces(showlegend=False)
            fig.add_traces(scatter_traces.data, rows=n_row, cols=n_col)
            fig.update_xaxes(title_text=pair[0], row=n_row, col=n_col)
            fig.update_yaxes(title_text=pair[1], row=n_row, col=n_col)
            # fig.update_yaxes(tickvals=cross_tab.index, type='category', row=n_row, col=n_col)
            # fig.add_traces(bar_traces.data, rows=n_row, cols=n_col)
        fig.update_traces(marker=dict(line=dict(width=0.5, color='white')))

    fig.update_layout(width=fig_width, height=fig_height, barmode='relative', title_text=title, title_x=0.5)
    fig.show(width=fig_width, height=fig_height)