import numpy as np
from sklearn import model_selection
from sklearn import metrics
import plotly.graph_objects as go

def get_pr_curve(model, X, y, cv=None, verbose:int=0):
    y_pred = model_selection.cross_val_predict(model, X, y, cv=cv, method='predict_proba')
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_pred[:,1])
    
    f1 = 2 * precision * recall / (precision + recall)
    idx = np.nanargmax(f1)
    if verbose >= 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, name='PR-curve', text=f1, fill='tozeroy'))
        fig.add_trace(go.Scatter(x=recall[[idx]], y=precision[[idx]], name='Best F1-Score', marker=dict(size=10), mode='markers', text=f1[[idx]]))
        fig.update_layout(title=dict(x=0.5, text='PR-curve', subtitle_text=f'Best threshold = {thresholds[idx]:.3f}, F1-Score = {f1[idx]:.3f}'),
                        width=800)
        fig.show()
    return thresholds[idx], f1[idx]