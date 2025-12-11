import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from scipy.stats import gaussian_kde

class OutlierCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'iqr', log: bool = False, left: float = 1.5, right: float = 1.5, mapping: dict | None = None,
                 handle: str = 'nan', verbose: int = 0):
        """
        Инициализация трансформера для удаления выбросов.

        Параметры:
        ----------
        method : str, default='iqr'
            Метод определения выбросов ('iqr' или 'zscore').
        log : bool, default=False
            Логарифмировать ли данные перед обработкой.
        left : float, default=1.5
            Левый множитель для метода IQR или граница для z-score.
        right : float, default=1.5
            Правый множитель для метода IQR или граница для z-score.
        mapping : dict, default=None
            Словарь с настройками для конкретных столбцов.
            Пример: {'col1': {'method': 'iqr', 'log': True, 'left': 1.5, 'right': 1.5}}
        handle : str, default='nan'
            Действие над аномальными значениями: 'nan' - замена аномальных значений на np.nan, 'drop' - удаление строки с аномальным значением.
        verbose : int, default=0
            При verbose=1 во время вызова fit выводится диаграмма распределения с границами определения выбросов
        """
        self.method = method
        self.log = log
        self.left = left
        self.right = right
        self.mapping = mapping if mapping is not None else {}
        self.handle = handle
        self.verbose = verbose
        self.bounds_ = {}  # Здесь будут храниться границы для каждого столбца

    def fit(self, X, y=None):
        """
        Вычисляет границы выбросов для каждого столбца.

        Параметры:
        ----------
        X : array-like, DataFrame
            Входные данные.
        y : None
            Не используется, здесь для совместимости.

        Возвращает:
        -----------
        self
        """
        X = pd.DataFrame(X)  # Для удобства работы

        if self.verbose >= 1:
            cols = self.mapping.keys()
            n_cols = 3
            n_rows = math.ceil(len(cols) / 3)
            fig = make_subplots(rows=n_rows, cols=n_cols, vertical_spacing=0.08)

        for i, col in enumerate(self.mapping):
            # Получаем настройки для текущего столбца или используем значения по умолчанию
            col_params: dict = self.mapping.get(col, {})
            method = col_params.get('method', self.method)
            log = col_params.get('log', self.log)
            left = col_params.get('left', self.left)
            right = col_params.get('right', self.right)

            data: pd.Series = X[col].copy()

            if log:
                if (data <= 0).any():
                    raise ValueError(f'Столбец {col}. При логарифмическом масштабировании в данных не должны содержаться значения меньше 0')
                data = np.log1p(data)  # Логарифмируем (с учетом нулей)

            if method == 'iqr':
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - left * iqr
                upper_bound = q3 + right * iqr
            elif method == 'zscore':
                mean = data.mean()
                std = data.std()
                lower_bound = mean - left * std
                upper_bound = mean + right * std
            else:
                raise ValueError(f"Неизвестный метод: {method}")

            # Сохраняем границы (и параметры, чтобы правильно преобразовать в transform)
            self.bounds_[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'log': log,
                'method': method,
                'left': left,
                'right': right
            }

            if self.verbose >= 1:
                n_row = i // 3 + 1
                n_col = i % 3 + 1
                fig.add_trace(go.Histogram(x=data, xbins=dict(size=(data.max()-data.min()) / 50)), row=n_row, col=n_col)
                fig.add_vline(x=lower_bound, row=n_row, col=n_col)
                fig.add_vline(x=upper_bound, row=n_row, col=n_col)
                fig.layout['annotations'][i]['text'] = f'{col}<br>Процент выбросов: {data[(data >= lower_bound) & (data <= upper_bound)].shape[0] / data.shape[0] * 100:.2f}%'

        if self.verbose >= 1:
            fig.update_layout(showlegend=False, width=1600)
            fig.show()

        return self

    def transform(self, X, y):
        """
        Удаляет выбросы на основе границ, вычисленных в fit.

        Параметры:
        ----------
        X : array-like, DataFrame
            Входные данные.

        Возвращает:
        -----------
        DataFrame без выбросов (заменены на NaN).
        """
        X = pd.DataFrame(X)
        
        outliers_idx = [] 
        for col in self.bounds_:

            bounds = self.bounds_[col]
            lower_bound = bounds['lower_bound']
            upper_bound = bounds['upper_bound']
            log = bounds['log']

            data: pd.Series = X[col].copy()

            if log:
                data = np.log1p(data)

            # Заменяем выбросы на NaN
            if self.handle == 'nan':
                data[(data < lower_bound) | (data > upper_bound)] = np.nan
            elif self.handle == 'drop':
                outliers_idx += list(X[(data >= lower_bound) & (data <= upper_bound)].index)
            else:
                raise ValueError('Неизвестное действие:', self.handle)

            if log:
                data = np.expm1(data)  # Обратное преобразование

            X[col] = data

        return X


class LowInformationCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, top_freq_thresh: float = 0.99, nunique_ratio_thresh: float = 0.99, verbose:int=0):
        self.top_freq_thresh = top_freq_thresh
        self.nunuqie_ratio_thresh = nunique_ratio_thresh
        self.verbose = verbose
        self.cols_to_drop = []
        

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        
        self.cols_to_drop = []
        for col in X.columns:
            top_freq = X[col].value_counts(normalize=True).max() 
            nunique_ratio = X[col].nunique() / X[col].shape[0]
            if top_freq > self.top_freq_thresh or nunique_ratio > self.nunuqie_ratio_thresh:
                if self.verbose >= 1:
                    print(f'{col} - {top_freq*100:.3f}% одинаковых значений, {nunique_ratio*100:.3f}% уникальных значений')
                self.cols_to_drop.append(col)

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.drop(columns=self.cols_to_drop)
    
    
class MulticollinearityCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=0.7, method='pearson', verbose: int=0, plt_width=1000, plt_height=800):
        self.cols_to_drop = []
        self.thresh=thresh
        self.method=method
        self.verbose = verbose
        self.plt_width=plt_width
        self.plt_height=plt_height
        self.high_corr = pd.Series()
        
    
    def fit(self, X, y=None):
        # Отрисовка тепловой карты корреляции и вывод талибцы с парами с абсолютными значениями корреляции > thresh
        if y is not None:
            self.corr = pd.concat([X, y], axis=1).corr(numeric_only=True, method=self.method)
            X_corr = self.corr.drop(index=y.name, columns=y.name)
            self.y_corr = self.corr[y.name].drop(index=y.name).sort_values(ascending=False, key=abs)
        else:
            X_corr = X.corr(numeric_only=True, method=self.method)
        if self.verbose >= 2:
            px.imshow(X_corr.round(2), width=self.plt_width, height=self.plt_height, zmin=-1, zmax=1,
                    color_continuous_scale=px.colors.diverging.BrBG, text_auto=True, title='Корреляция признаков').update_layout(title_x=0.5).show()
            if y is not None:
                y_corr_data = pd.concat([self.y_corr.apply(abs), pd.Series(
                    ['Прямая' if val > 0 else 'Обратная' for val in self.y_corr], name='Корреляция', index=self.y_corr.index)], axis=1).sort_values(by=y.name)
                fig = go.Figure(data=[
                    go.Bar(
                        y=y_corr_data.index,
                        x=y_corr_data[y.name],
                        marker_color=y_corr_data['Корреляция'].apply(lambda x: px.colors.diverging.BrBG[-3] if x=='Прямая' else px.colors.diverging.BrBG[2]),
                        orientation='h'
                    )
                ])
                fig.update_layout(width=self.plt_width, height=self.plt_height, title_text=f'Корреляция факторов с целевым признаком {y.name}', title_x=0.5, yaxis_tickvals=y_corr_data.index, yaxis_type='category')
                fig.show()
        corr_pairs = X_corr.stack().reset_index()
        corr_pairs.columns = ['feature 1', 'feature 2', 'correlation']
        corr_pairs = corr_pairs[corr_pairs['feature 1'] < corr_pairs['feature 2']]  # Удаление дубликатов
        self.high_corr: pd.Series = corr_pairs[corr_pairs['correlation'].abs() > self.thresh]
        
        if y is not None:
            if not self.high_corr.empty:
                self.cols_to_drop = list(self.high_corr.apply(lambda x: x['feature 1'] if abs(self.y_corr[x['feature 1']]) < abs(self.y_corr[x['feature 2']]) else x['feature 2'], axis=1))
                if self.verbose >= 1:
                    print('Столбцы к удалению:', self.cols_to_drop)
    
    
    def transform(self, X):
        X = pd.DataFrame(X)
        return X.drop(columns=self.cols_to_drop)

class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, method='classic'):
        self.method = method
        self.cols = cols
        
        self.nunique_cols = pd.Series()
        self.cols_to_label_encoding = [] 
        self.cols_to_oh_encoding = []
        self.cols_to_bin_encoding =  []
        
        self.cols_droped = []
        self.cols_added = []
        
        if self.method not in ['classic', 'ordinal']: 
            raise ValueError('Неизвестный метод:', method)
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        self.nunique_cols = X[self.cols].nunique()
        if self.method == 'classic':
            self.cols_to_label_encoding = list(self.nunique_cols[self.nunique_cols == 2].index)
            self.cols_to_oh_encoding = list(self.nunique_cols[(self.nunique_cols > 2) & (self.nunique_cols <= 15)].index)
            self.cols_to_bin_encoding = list(self.nunique_cols[self.nunique_cols > 15].index)
            
            self.label_encoder = LabelEncoder()
            self.oh_encoder = ce.OneHotEncoder(use_cat_names=True)
            self.bin_encoder = ce.BinaryEncoder()
            
            
        elif self.method == 'ordinal':
            order_values = {}
            for col in self.cols:
                order_values[col] = list(pd.crosstab(index=X[col].astype(str), columns=y, normalize='index').sort_values(by=list(y.unique())).index)
            self.ord_encoder = ce.OrdinalEncoder(mapping=[{'col': col, 'mapping': {val: i for i, val in enumerate(order_values[col])}} for col in order_values])

    def transform(self, X):
        X = pd.DataFrame(X)
        X[self.cols] = X[self.cols].astype(str)
        if self.method == 'classic':
            # Label Encoding
            for col in self.cols_to_label_encoding:
                X[col] = self.label_encoder.fit_transform(X[col])
            
            # One-Hot Encoding
            ohe_data = self.oh_encoder.fit_transform(X[self.cols_to_oh_encoding])
            X = pd.concat([X, ohe_data], axis=1)
            X.drop(columns=self.cols_to_oh_encoding, inplace=True)
            cols_droped += self.cols_to_oh_encoding
            cols_added += list(self.oh_encoder.get_feature_names_out())
                
            # Binary Encoding
            bine_data = self.bin_encoder.fit_transform(X[self.cols_to_bin_encoding].astype('object'))
            X = pd.concat([X, bine_data], axis=1)
            X.drop(columns=self.cols_to_bin_encoding, inplace=True)
            cols_droped += self.cols_to_bin_encoding
            cols_added += list(self.bin_encoder.get_feature_names_out())
        elif self.method == 'ordinal':
            X[self.cols] = self.ord_encoder.fit_transform(X[self.cols])
        
        return X
    
    
def clean_outliers(data: pd.Series, method='z', log=False, left=3, right=3, kde_plt=True):
    data = data.copy()
    if log:
        data_min = data.min()
        data: pd.Series = np.log(data - data_min + 1)
    if method == 'z':
        mu = data.mean()
        sigma = data.std()
        left_bound = mu - left * sigma
        right_bound = mu + right * sigma
    elif method == 'iqr':
        quartile_1, quartile_3 = data.quantile(0.25), data.quantile(0.75),
        iqr = quartile_3 - quartile_1
        left_bound = quartile_1 - (iqr * left)
        right_bound = quartile_3 + (iqr * right)
    elif method == 'define':
        left_bound = left
        right_bound = right

    outliers = data[(data < left_bound) | (data > right_bound)]
    cleaned = data[(data >= left_bound) & (data <= right_bound)]

    # Диаграмма
    fig = make_subplots(rows=1, cols=2, subplot_titles=['До' + (' (log)'if log else ''), 'После'])
    for i, d in enumerate([data, cleaned]):
        fig.add_trace(go.Histogram(x=d, xbins=dict(size=(d.max()-d.min()) / 50)), row=1, col=i+1)
        # KDE
        if kde_plt:
            kde = gaussian_kde(d)
            x_kde = np.linspace(d.min(), d.max(), 500)
            y_kde_density = kde(x_kde)
            y_kde_counts = (y_kde_density - np.min(y_kde_density)) / (np.max(y_kde_density) - np.min(y_kde_density)) * max(np.histogram(d, bins=50)[0])
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
                ), row=1, col=i+1)
        
        fig.update_xaxes(title_text="Значение", row=1, col=i+1)
        fig.update_yaxes(title_text="Плотность", row=1, col=i+1)
        if i == 0:
            fig.add_vline(x=left_bound, row=1, col=i+1)
            fig.add_vline(x=right_bound, row=1, col=i+1)
    fig.update_layout(showlegend=False, width=1600,
                      title=dict(x=0.5, text=data.name,
                                 subtitle_text=f'Процент выбросов: {outliers.shape[0] / data.shape[0] * 100:.2f}% | '
                                 + f'Границы выбросов - [{round(np.exp(left_bound)+data_min-1 if log else left_bound, 3)}:{round(np.exp(right_bound)+data_min-1 if log else right_bound, 3)}]'))
    fig.show()

    if log:
        outliers = np.exp(outliers) + data_min - 1
        cleaned = np.exp(cleaned) + data_min - 1

    return outliers, cleaned
