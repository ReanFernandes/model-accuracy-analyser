import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import curve_fit
# import plotly.figure_factory as ff # Not used, removed
import json
import traceback # Keep for debugging if needed
from collections.abc import Iterable # Import Iterable

# Set page config
st.set_page_config(layout="wide", page_title="Model Performance Analysis")

# --- Helper Functions ---

def get_numeric_columns(df):
    """Returns a list of numeric column names, excluding boolean."""
    if df is None:
        return []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    bool_cols = df.select_dtypes(include=[np.bool_, 'boolean']).columns.tolist()
    return [col for col in numeric_cols if col not in bool_cols]

def extract_set_type(dataset):
    """Helper to determine if dataset is test or validation."""
    if pd.isna(dataset):
        return 'unknown'
    dataset_str = str(dataset).lower() # Ensure string and lowercase
    if 'test_set' in dataset_str:
        return 'test'
    elif 'val_set' in dataset_str:
        return 'validation'
    return 'other'

def get_set_number(dataset):
    """Helper to extract the number from dataset string."""
    if pd.isna(dataset):
        return np.nan
    try:
        # Extract digits, handle cases like 'set_1' or 'set1'
        digits = ''.join(filter(str.isdigit, str(dataset)))
        return int(digits) if digits else np.nan
    except (ValueError, TypeError):
        return np.nan

def clean_label(text):
    """Make column names and labels more readable."""
    if pd.isna(text) or text is None or text == "None":
        return ""
    # Replace underscores with spaces
    clean_text = str(text).replace('_', ' ')
    # Capitalize each word
    clean_text = ' '.join(word.capitalize() for word in clean_text.split())
    return clean_text

def sort_data_by_column(df, column, sort_order="Ascending"):
    """Sort dataframe by a column in specified order."""
    if sort_order == "No Sorting" or column is None or column == "None" or column not in df.columns:
        return df

    df_copy = df.copy() # Work on a copy

    # Attempt numeric sort first
    numeric_col = pd.to_numeric(df_copy[column], errors='coerce')
    if not numeric_col.isna().all(): # If at least some values are numeric
        df_copy['_sort_val_'] = numeric_col
        return df_copy.sort_values(by='_sort_val_', ascending=(sort_order == "Ascending"), na_position='last').drop('_sort_val_', axis=1)
    else:
        # Fallback to string sorting, handle potential mixed types
        try:
            # Ensure all values are strings before sorting
            return df_copy.sort_values(by=column, key=lambda col: col.astype(str), ascending=(sort_order == "Ascending"), na_position='last')
        except TypeError: # Final fallback if conversion fails
             return df_copy.sort_values(by=column, ascending=(sort_order == "Ascending"), na_position='last')


# --- Plotting & Analysis Functions ---

def power_law(x, a, b, c):
    """Power law function for fitting learning curves: y = a - b * x^(-c)."""
    x_numeric = pd.to_numeric(x, errors='coerce')
    # Replace NaNs or non-positives with a small epsilon
    x_safe = np.where(np.isnan(x_numeric) | (x_numeric <= 0), 1e-10, x_numeric)
    # Avoid potential overflow or invalid values in power calculation
    with np.errstate(over='ignore', invalid='ignore'):
        power_term = np.power(x_safe, -c)
        # Replace inf or nan results from power term with a large number or zero
        power_term = np.nan_to_num(power_term, nan=0.0, posinf=1e10) # Use 1e10 as large number
    return a - b * power_term

# MODIFICATION: Accept group_col as parameter
def analyze_learning_curve(data, group_col='model_name'):
    """
    Analyze learning curves for each group in the dataset.
    Assumes 'num_training_samples', 'label_accuracy', and the specified group_col columns.
    """
    required_cols = [group_col, 'num_training_samples', 'label_accuracy']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Missing required columns for learning curve analysis: Need {', '.join(required_cols)}")
        return pd.DataFrame(), {}, {}

    data = data.copy()
    data['num_training_samples'] = pd.to_numeric(data['num_training_samples'], errors='coerce')
    data['label_accuracy'] = pd.to_numeric(data['label_accuracy'], errors='coerce')
    data = data.dropna(subset=['num_training_samples', 'label_accuracy', group_col])

    if data.empty:
        st.warning("No valid data for learning curve analysis after cleaning.")
        return pd.DataFrame(), {}, {}

    # Group by the specified group_col and sample size
    grouped = data.groupby([group_col, 'num_training_samples'])['label_accuracy'].agg(
        median='median', mean='mean', std='std',
        q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), count='count'
    ).reset_index()

    groups = data[group_col].unique()
    fit_params = {}
    r_squared = {}

    for group in groups:
        model_data = grouped[grouped[group_col] == group].sort_values('num_training_samples')
        if len(model_data) < 3: continue

        x = model_data['num_training_samples'].values
        y = model_data['median'].values
        valid_mask = (x > 0) & (~np.isnan(y)) # Ensure x is positive and y is not NaN
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) < 3: continue

        try:
            # Define bounds more carefully based on data range
            y_min_data, y_max_data = y_valid.min(), y_valid.max()
            a_upper_bound = max(1.0, y_max_data * 1.5) # Allow asymptote above max observed
            a_lower_bound = min(0.0, y_min_data * 0.5) # Allow asymptote below min observed
            p0 = [y_max_data, 0.1, 0.1] # Adjust initial guess
            bounds = ([a_lower_bound, 0, 1e-9], [a_upper_bound, np.inf, np.inf]) # Bounds for a, b, c (ensure c > 0)

            popt, pcov = curve_fit(power_law, x_valid, y_valid, p0=p0, bounds=bounds, maxfev=5000, method='trf')

            fit_params[group] = popt
            y_pred = power_law(x_valid, *popt)
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r_squared[group] = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else 1.0 # Avoid division by zero

        except (RuntimeError, ValueError, Exception) as e: # Catch specific and general errors
            # st.warning(f"Could not fit curve for {group}: {e}") # Reduce verbosity
            fit_params[group] = None # Explicitly mark as failed
            r_squared[group] = None

    return grouped, fit_params, r_squared

def plot_learning_curves_plotly(grouped, fit_params, r_squared, human_baseline=0.675, random_chance=0.25, use_log_x=True, show_error_bars=True, colors=px.colors.qualitative.Plotly, group_col='model_name'):
    """Create a Plotly figure of learning curves."""
    fig = go.Figure()
    # Ensure group_col exists before proceeding
    if group_col not in grouped.columns:
         st.error(f"Grouping column '{group_col}' not found in the grouped learning curve data.")
         return fig # Return empty figure

    models = grouped[group_col].unique()
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}

    all_x = grouped['num_training_samples'].unique()
    min_x_overall = all_x[all_x > 0].min() if any(all_x > 0) else 1
    max_x_overall = all_x.max() if len(all_x) > 0 else 100
    min_y_overall = grouped['median'].min() if not grouped.empty else 0
    max_y_overall = grouped['median'].max() if not grouped.empty else 1

    for model in models:
        model_data = grouped[grouped[group_col] == model].sort_values('num_training_samples')
        if len(model_data) < 1: continue

        error_y_config = None
        if show_error_bars:
             error_y_config=dict(type='data', symmetric=False, array=model_data['q75'] - model_data['median'], arrayminus=model_data['median'] - model_data['q25'], thickness=1, width=3)

        fig.add_trace(go.Scatter(
            x=model_data['num_training_samples'], y=model_data['median'],
            error_y=error_y_config, mode='markers+lines' if len(model_data) > 1 else 'markers',
            name=f"{str(model)} (Median)", marker=dict(color=model_colors.get(model, 'grey'), size=8), # Use .get for safety
            line=dict(color=model_colors.get(model, 'grey'), width=1.5), legendgroup=str(model)
        ))

        if model in fit_params and fit_params[model] is not None: # Check for successful fit
            params = fit_params[model]
            r2 = r_squared.get(model)
            r2_text = f", R²={r2:.3f}" if r2 is not None else ""

            min_x_fit = model_data['num_training_samples'][model_data['num_training_samples'] > 0].min() if any(model_data['num_training_samples'] > 0) else min_x_overall
            max_x_fit = model_data['num_training_samples'].max() * 1.2
            min_x_fit_safe = max(min_x_fit, 1e-9) # Ensure positive for logspace

            if use_log_x:
                 x_smooth = np.logspace(np.log10(min_x_fit_safe), np.log10(max_x_fit), 100)
            else:
                x_smooth = np.linspace(min_x_fit, max_x_fit, 100)

            x_smooth_safe = x_smooth[x_smooth > 0]
            if len(x_smooth_safe) > 0:
                y_smooth = power_law(x_smooth_safe, *params)
                fig.add_trace(go.Scatter(
                    x=x_smooth_safe, y=y_smooth, mode='lines',
                    name=f"{str(model)} (Fit{r2_text})",
                    line=dict(color=model_colors.get(model, 'grey'), dash='dash', width=2),
                    legendgroup=str(model)
                ))

    fig.add_hline(y=human_baseline, line=dict(color="purple", width=2, dash="dash"), annotation_text="Human Baseline", annotation_position="top right")
    fig.add_hline(y=random_chance, line=dict(color="gray", width=2, dash="dash"), annotation_text="Random Chance", annotation_position="bottom right")

    xaxis_config = dict(title="Number of Training Samples", showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')
    if use_log_x:
        xaxis_config['type'] = "log"; xaxis_config['exponentformat'] = "power"
        min_x_range = max(min_x_overall, 1e-9)
        xaxis_config['range'] = [np.log10(min_x_range), np.log10(max_x_overall * 1.5)]
    else:
        xaxis_config['type'] = "linear"; xaxis_config['range'] = [0, max_x_overall * 1.1]

    fig.update_layout(
        title="Learning Curve: Accuracy vs. Number of Training Samples", xaxis=xaxis_config,
        yaxis=dict(title="Median Label Accuracy", range=[0, 1], showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)'),
        legend=dict(title=clean_label(group_col), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="rgba(0, 0, 0, 0.2)", borderwidth=1),
        hovermode="closest", template="plotly_white", margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# MODIFICATION: Accept group_col as parameter
def calculate_efficiency_metrics(grouped, fit_params, group_col='model_name'):
    """Calculate various efficiency metrics."""
    metrics = {}
    # Ensure group_col exists
    if group_col not in grouped.columns:
         st.error(f"Grouping column '{group_col}' not found for efficiency metrics calculation.")
         return metrics

    groups = grouped[group_col].unique()

    for group in groups:
        model_data = grouped[grouped[group_col] == group].sort_values('num_training_samples')
        if len(model_data) < 2: continue

        min_positive_samples_data = model_data[model_data['num_training_samples'] > 0]
        if min_positive_samples_data.empty: continue # Skip if no positive samples
        baseline_row = min_positive_samples_data.iloc[0]
        baseline_x = baseline_row['num_training_samples']
        baseline_y = baseline_row['median']

        max_accuracy_row = model_data.loc[model_data['median'].idxmax()]
        max_accuracy = max_accuracy_row['median']
        max_x = model_data['num_training_samples'].max()

        total_improvement = max_accuracy - baseline_y
        sample_efficiency = 0.0
        if max_x > 0 and baseline_x > 0 and max_x > baseline_x:
            log_diff = np.log10(max_x) - np.log10(baseline_x)
            if log_diff > 1e-9: sample_efficiency = total_improvement / log_diff
            else: sample_efficiency = np.inf
        elif max_x > 0 and baseline_x <= 0: sample_efficiency = total_improvement / np.log10(max_x + 1)

        plateau_samples, plateau_accuracy = None, None
        improvement_threshold = 0.01
        for i in range(1, len(model_data)):
            curr_row, prev_row = model_data.iloc[i], model_data.iloc[i-1]
            curr_x, prev_x = curr_row['num_training_samples'], prev_row['num_training_samples']
            curr_y, prev_y = curr_row['median'], prev_row['median']
            if prev_y is not None and prev_y != 0 and curr_x > prev_x:
                rel_improvement = (curr_y - prev_y) / abs(prev_y)
                if rel_improvement < improvement_threshold:
                    plateau_samples, plateau_accuracy = prev_x, prev_y
                    break

        metrics[group] = {
            'baseline_accuracy': baseline_y, 'max_accuracy': max_accuracy,
            'total_improvement': total_improvement,
            'relative_improvement': (total_improvement / abs(baseline_y) * 100) if baseline_y != 0 else np.inf,
            'sample_efficiency': sample_efficiency, 'plateau_samples': plateau_samples,
            'plateau_accuracy': plateau_accuracy
        }

        if total_improvement > 0:
            for _, row in model_data[model_data['num_training_samples'] >= baseline_x].iterrows():
                sample_size, current_gain = row['num_training_samples'], row['median'] - baseline_y
                pct_of_max_gain = (current_gain / total_improvement) * 100
                try: x_label = int(sample_size)
                except: x_label = str(sample_size)
                metrics[group][f'pct_gain_at_{x_label}'] = pct_of_max_gain
        else:
             for _, row in model_data[model_data['num_training_samples'] >= baseline_x].iterrows():
                 try: x_label = int(row['num_training_samples'])
                 except: x_label = str(row['num_training_samples'])
                 metrics[group][f'pct_gain_at_{x_label}'] = 0.0
    return metrics

def add_sample_counts_to_figure(fig, df, x_col, y_col, color_col=None):
    """Add sample count annotations."""
    if x_col not in df.columns or y_col not in df.columns: return fig
    df_copy = df.copy()
    df_copy[y_col] = pd.to_numeric(df_copy[y_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[x_col, y_col])
    if df_copy.empty: return fig

    try:
        group_cols = [x_col]
        if color_col and color_col != "None" and color_col in df_copy.columns:
            group_cols.append(color_col)

        counts = df_copy.groupby(group_cols).size().reset_index(name='count')
        max_y_values = df_copy.groupby(group_cols)[y_col].max().reset_index()
        counts = pd.merge(counts, max_y_values, on=group_cols, how='left')

        # Fallback for positioning if max_y fails for a group
        overall_max_y = df_copy.groupby(x_col)[y_col].max().reset_index().rename(columns={y_col: '_overall_max_y'})
        counts = pd.merge(counts, overall_max_y, on=x_col, how='left')
        counts[y_col] = counts[y_col].fillna(counts['_overall_max_y'])
        counts = counts.drop(columns=['_overall_max_y']).dropna(subset=[y_col])

        for _, row in counts.iterrows():
            fig.add_annotation(x=row[x_col], y=row[y_col], text=f'n={row["count"]}', showarrow=False, yshift=10, font=dict(size=9, color='grey'))
    except Exception as e:
        st.warning(f"Could not add sample count annotations: {e}")
    return fig

def create_and_update_plot(fig, df, x_label, y_label, color_label, facet_row, facet_col, plot_type, key_suffix, current_colors):
    """Update plot with formatting and display it (enhanced UI)."""
    clean_x_label = clean_label(x_label)
    clean_y_label = clean_label(y_label)
    clean_color_label = clean_label(color_label) if color_label and color_label != "None" else ""
    clean_facet_row = clean_label(facet_row) if facet_row and facet_row != "None" else ""
    clean_facet_col = clean_label(facet_col) if facet_col and facet_col != "None" else ""

    default_title = f"{clean_y_label} vs {clean_x_label}"
    if clean_color_label: default_title += f" by {clean_color_label}"
    if clean_facet_row or clean_facet_col:
        facet_dims = [f"Rows: {clean_facet_row}"] if clean_facet_row else []
        if clean_facet_col: facet_dims.append(f"Cols: {clean_facet_col}")
        default_title += f" ({', '.join(facet_dims)})"

    # Initialize customization variables with defaults
    custom_settings = {}

    with st.expander("🎨 Customize Plot Appearance", expanded=False):
        cust_tabs = st.tabs(["Layout & Axes", "Colors & Style", "Reference Lines", "Annotations"])

        with cust_tabs[0]:
            st.markdown("#### General Layout")
            col1, col2 = st.columns(2)
            with col1:
                custom_settings['plot_title'] = st.text_input("Plot Title", value=default_title, key=f"title_{key_suffix}")
                custom_settings['title_font_size'] = st.slider("Title Font Size", 10, 40, 18, key=f"title_font_{key_suffix}")
                custom_settings['plot_height'] = st.slider("Plot Height (px)", 300, 1500, 600, 50, key=f"height_{key_suffix}")
            with col2:
                custom_settings['show_legend'] = st.checkbox("Show Legend", True, key=f"legend_{key_suffix}")
                custom_settings['legend_font_size'] = st.slider("Legend Font Size", 6, 24, 10, key=f"legend_size_{key_suffix}", disabled=not custom_settings['show_legend'])
                custom_settings['legend_title_size'] = st.slider("Legend Title Size", 8, 26, 12, key=f"legend_title_size_{key_suffix}", disabled=not custom_settings['show_legend'])

            st.markdown("#### Axes Configuration")
            col1, col2 = st.columns(2)
            with col1:
                custom_settings['axis_title_size'] = st.slider("Axis Title Size", 8, 30, 14, key=f"axis_title_size_{key_suffix}")
                custom_settings['axis_tick_size'] = st.slider("Axis Tick Size", 6, 24, 12, key=f"axis_tick_size_{key_suffix}")
                custom_settings['y_tick_step'] = st.number_input("Y-axis Tick Step Size", min_value=0.0, max_value=10.0, value=0.1, step=0.01, help="Interval between y-axis tick marks. Set to 0 for auto.", key=f"y_tick_step_{key_suffix}")

            with col2:
                custom_settings['y_range_auto'] = st.checkbox("Auto Y-axis Range", True, key=f"y_auto_{key_suffix}")
                y_min_val, y_max_val = 0.0, 1.0
                # Ensure y_label exists and is numeric before calculating min/max
                if y_label in df.columns and pd.api.types.is_numeric_dtype(df[y_label]):
                    y_numeric = df[y_label].dropna()
                    if not y_numeric.empty: y_min_val, y_max_val = float(y_numeric.min()), float(y_numeric.max())
                custom_settings['y_min'] = st.number_input("Y-axis Min", value=y_min_val, key=f"ymin_{key_suffix}", disabled=custom_settings['y_range_auto'])
                custom_settings['y_max'] = st.number_input("Y-axis Max", value=y_max_val, key=f"ymax_{key_suffix}", disabled=custom_settings['y_range_auto'])
                custom_settings['y_range'] = None if custom_settings['y_range_auto'] else [custom_settings['y_min'], custom_settings['y_max']]

                custom_settings['x_range_auto'] = st.checkbox("Auto X-axis Range", True, key=f"x_auto_{key_suffix}")
                x_min_default, x_max_default = 0.0, 1.0
                # Ensure x_label exists and is numeric before calculating min/max
                if x_label in df.columns and pd.api.types.is_numeric_dtype(df[x_label]):
                    x_numeric = df[x_label].dropna()
                    if not x_numeric.empty: x_min_default, x_max_default = float(x_numeric.min()), float(x_numeric.max())
                custom_settings['x_min'] = st.number_input("X-axis Min", value=x_min_default, key=f"xmin_{key_suffix}", disabled=custom_settings['x_range_auto'])
                custom_settings['x_max'] = st.number_input("X-axis Max", value=x_max_default, key=f"xmax_{key_suffix}", disabled=custom_settings['x_range_auto'])
                custom_settings['x_range'] = None if custom_settings['x_range_auto'] else [custom_settings['x_min'], custom_settings['x_max']]

        with cust_tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                custom_settings['theme'] = st.selectbox("Plot Theme", ["plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white"], index=0, key=f"theme_{key_suffix}")
                custom_settings['show_grid'] = st.checkbox("Show Grid Lines", True, key=f"grid_{key_suffix}")
                if plot_type == 'box':
                    st.markdown("**Box Plot Options**")
                    custom_settings['box_points'] = st.selectbox("Show Points", ["outliers", "all", "suspectedoutliers", False], index=0, key=f"boxpoints_{key_suffix}")
                    custom_settings['box_notched'] = st.checkbox("Notched Boxes", False, key=f"boxnotch_{key_suffix}")
                    custom_settings['box_mode'] = st.selectbox("Box Mode", ["group", "overlay"], index=0, key=f"boxmode_{key_suffix}")

            with col2:
                color_options = {name: getattr(px.colors.qualitative, name) for name in px.colors.qualitative.__all__}
                color_options.update({name: getattr(px.colors.sequential, name) for name in px.colors.sequential.__all__ if not name.endswith("_r")})
                selected_palette_name = st.selectbox("Color Palette", list(color_options.keys()), index=list(color_options.keys()).index('Plotly') if 'Plotly' in color_options else 0, key=f"palette_{key_suffix}")
                selected_color_list = color_options.get(selected_palette_name, current_colors) # Fallback
                custom_settings['active_colors'] = selected_color_list

                st.write("Color Preview:")
                # FIX: Check if selected_color_list is iterable and not a function
                if isinstance(selected_color_list, Iterable) and not callable(selected_color_list):
                    color_preview_html = "".join([f'<span title="{color}" style="display:inline-block;width:20px;height:20px;background:{color};margin:1px;border:1px solid lightgrey;"></span>' for color in selected_color_list])
                    st.markdown(color_preview_html, unsafe_allow_html=True)
                else:
                     st.warning(f"Selected color palette '{selected_palette_name}' is not a valid list of colors.")
                     custom_settings['active_colors'] = current_colors # Revert

        with cust_tabs[2]: # Reference Lines
            st.markdown("Define reference lines (horizontal or vertical). Use 'mean' or 'median' as value to calculate from data.")
            ref_line_session_key = f"ref_lines_{key_suffix}"
            if ref_line_session_key not in st.session_state: st.session_state[ref_line_session_key] = []
            ref_lines_list = st.session_state[ref_line_session_key]
            indices_to_remove = []
            if not ref_lines_list: st.caption("No reference lines defined yet.")
            for i, line in enumerate(ref_lines_list):
                cols = st.columns([1, 2, 2, 1, 1, 1, 1])
                cols[0].text_input("Axis", value=line.get('axis','y').upper(), key=f"ref_axis_disp_{key_suffix}_{i}", disabled=True)
                cols[1].text_input("Value/Stat", value=str(line.get('value', '')), key=f"ref_val_disp_{key_suffix}_{i}", disabled=True)
                cols[2].text_input("Label", value=line.get('label', ''), key=f"ref_lab_disp_{key_suffix}_{i}", disabled=True)
                cols[3].markdown(f"<span style='background-color:{line.get('color', '#000000')}; padding: 2px 5px; border-radius: 3px; color: white; mix-blend-mode: difference;'>{line.get('color', '#000000')}</span>", unsafe_allow_html=True)
                cols[4].text_input("Style", value=line.get('style', 'dash'), key=f"ref_sty_disp_{key_suffix}_{i}", disabled=True)
                cols[5].text_input("Width", value=str(line.get('width', 1)), key=f"ref_wid_disp_{key_suffix}_{i}", disabled=True)
                if cols[6].button("❌", key=f"ref_rem_{key_suffix}_{i}", help="Remove this line"): indices_to_remove.append(i)
            if indices_to_remove:
                 for i in sorted(indices_to_remove, reverse=True): ref_lines_list.pop(i)
                 st.rerun()
            st.markdown("---"); st.markdown("##### Add New Reference Line")
            cols_new = st.columns([1, 2, 2, 1, 1, 1, 1])
            new_axis = cols_new[0].selectbox("Axis", ['y', 'x'], key=f"ref_axis_new_{key_suffix}")
            new_value = cols_new[1].text_input("Value or Stat ('mean', 'median')", key=f"ref_val_new_{key_suffix}")
            new_label = cols_new[2].text_input("Label (optional)", key=f"ref_lab_new_{key_suffix}")
            new_color = cols_new[3].color_picker("Color", "#888888", key=f"ref_col_new_{key_suffix}")
            new_style = cols_new[4].selectbox("Style", ['solid', 'dash', 'dot', 'dashdot'], index=1, key=f"ref_sty_new_{key_suffix}")
            new_width = cols_new[5].number_input("Width", min_value=1, max_value=10, value=1, step=1, key=f"ref_wid_new_{key_suffix}")
            if cols_new[6].button("➕ Add", key=f"ref_add_{key_suffix}"):
                if new_value:
                    is_stat = new_value.lower() in ['mean', 'median']
                    numeric_val = None
                    if not is_stat:
                        try: numeric_val = float(new_value)
                        except ValueError: st.error(f"Invalid value '{new_value}'. Must be numeric or 'mean'/'median'."); new_value = None
                    if new_value:
                         ref_lines_list.append({'axis': new_axis, 'value': numeric_val if not is_stat else new_value.lower(), 'label': new_label, 'color': new_color, 'style': new_style, 'width': new_width, 'position': 'top right'})
                         st.rerun()
                else: st.warning("Please provide a value or statistic.")

        with cust_tabs[3]:
            custom_settings['show_sample_counts'] = st.checkbox("Show Sample Counts (n=...)", value=False, key=f"show_counts_{key_suffix}")
            custom_settings['add_custom_annotation'] = st.checkbox("Add Custom Text Annotation", False, key=f"custom_annot_chk_{key_suffix}")
            if custom_settings['add_custom_annotation']:
                custom_settings['annot_text'] = st.text_input("Annotation Text", "My Note", key=f"annot_text_{key_suffix}")
                custom_settings['annot_x'] = st.number_input("X Position (numeric)", value=0.5, key=f"annot_x_{key_suffix}") # Simplified
                custom_settings['annot_y'] = st.number_input("Y Position", value=0.5, key=f"annot_y_{key_suffix}")
                custom_settings['annot_arrow'] = st.checkbox("Show Arrow", True, key=f"annot_arrow_{key_suffix}")
                custom_settings['annot_color'] = st.color_picker("Text Color", "#000000", key=f"annot_color_{key_suffix}")
                custom_settings['annot_size'] = st.slider("Text Size", 8, 24, 12, key=f"annot_size_{key_suffix}")

    # --- Apply Customizations & Display ---
    try:
        # Use .get with defaults for all custom_settings lookups
        fig.update_layout(template=custom_settings.get('theme', 'plotly_white'))
        fig.update_layout(
            title={'text': custom_settings.get('plot_title', default_title), 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=custom_settings.get('title_font_size', 18))},
            height=custom_settings.get('plot_height', 600),
            showlegend=custom_settings.get('show_legend', True),
            xaxis_title=clean_x_label, yaxis_title=clean_y_label,
            xaxis_range=custom_settings.get('x_range'), yaxis_range=custom_settings.get('y_range'),
            legend=dict(title=dict(text=clean_color_label, font=dict(size=custom_settings.get('legend_title_size', 12))), font=dict(size=custom_settings.get('legend_font_size', 10)), bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
            colorway=custom_settings.get('active_colors', current_colors), # Use fetched colors
            boxmode=custom_settings.get('box_mode', 'group') if plot_type == 'box' else None
        )
        fig.update_xaxes(showgrid=custom_settings.get('show_grid', True), title_font=dict(size=custom_settings.get('axis_title_size', 14)), tickfont=dict(size=custom_settings.get('axis_tick_size', 12)))
        # Handle y_tick_step=0 case for auto ticks
        y_dtick = custom_settings.get('y_tick_step', 0.1)
        fig.update_yaxes(showgrid=custom_settings.get('show_grid', True), title_font=dict(size=custom_settings.get('axis_title_size', 14)), tickfont=dict(size=custom_settings.get('axis_tick_size', 12)), dtick=y_dtick if y_dtick > 0 else None)

        if plot_type == 'box': fig.update_traces(boxpoints=custom_settings.get('box_points', 'outliers'), notched=custom_settings.get('box_notched', False), selector=dict(type='box'))

        active_ref_lines = st.session_state.get(f"ref_lines_{key_suffix}", [])
        if active_ref_lines:
            target_df_ref = df.copy()
            y_col_ref, x_col_ref = y_label, x_label
            if y_label == '_metric_value_' and '_metric_value_' in target_df_ref.columns: y_col_ref = '_metric_value_'
            if x_label == '_metric_variable_' and '_metric_variable_' in target_df_ref.columns: x_col_ref = '_metric_variable_'
            fig = add_reference_lines_generic(fig, target_df_ref, x_col_ref, y_col_ref, active_ref_lines)

        if custom_settings.get('show_sample_counts', False):
             y_col_counts = y_label if y_label in df.columns and pd.api.types.is_numeric_dtype(df[y_label]) else '_metric_value_' if '_metric_value_' in df.columns else None
             if y_col_counts: fig = add_sample_counts_to_figure(fig, df, x_label, y_col_counts, color_label)
             else: st.warning(f"Cannot add sample counts: Y-axis column '{y_label}' not valid.")

        if custom_settings.get('add_custom_annotation', False) and custom_settings.get('annot_text'):
            fig.add_annotation(
                x=custom_settings.get('annot_x', 0.5), y=custom_settings.get('annot_y', 0.5),
                text=custom_settings.get('annot_text', ''), showarrow=custom_settings.get('annot_arrow', True),
                arrowhead=1 if custom_settings.get('annot_arrow', True) else 0,
                font=dict(size=custom_settings.get('annot_size', 12), color=custom_settings.get('annot_color', "#000000")),
                align="left", bgcolor="rgba(255, 255, 255, 0.7)"
             )

    except Exception as e:
        st.error(f"Error applying plot customizations: {e}")
        st.code(traceback.format_exc())

    # Display Plot and Export Options
    st.plotly_chart(fig, use_container_width=True, key=f"plotly_{key_suffix}")
    st.markdown("---")
    st.markdown("##### Export Plot")
    export_cols = st.columns(3)
    plot_filename_base = f"{clean_y_label}_vs_{clean_x_label}"
    if clean_color_label: plot_filename_base += f"_by_{clean_color_label}"
    plot_filename_base = "".join(c if c.isalnum() else "_" for c in plot_filename_base).lower()

    try:
        png_bytes = pio.to_image(fig, format="png", scale=2)
        export_cols[0].download_button("Download PNG", png_bytes, f"{plot_filename_base}.png", "image/png", key=f"dl_png_{key_suffix}")
    except Exception as e: export_cols[0].warning(f"PNG export failed: {e}")
    try:
        html_buffer = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        export_cols[1].download_button("Download HTML", html_buffer, f"{plot_filename_base}.html", "text/html", key=f"dl_html_{key_suffix}")
    except Exception as e: export_cols[1].warning(f"HTML export failed: {e}")
    try:
        json_str = pio.to_json(fig)
        export_cols[2].download_button("Download JSON", json_str, f"{plot_filename_base}.json", "application/json", key=f"dl_json_{key_suffix}")
    except Exception as e: export_cols[2].warning(f"JSON export failed: {e}")


def add_reference_lines_generic(fig, df, x_col, y_col, ref_lines_config):
    """Adds reference lines (h or v) based on config (generic version)."""
    if not ref_lines_config or df is None or df.empty: return fig
    for line_config in ref_lines_config:
        axis = line_config.get('axis', 'y')
        value = line_config.get('value')
        label = line_config.get('label', '')
        color = line_config.get('color', 'grey')
        style = line_config.get('style', 'dash')
        width = line_config.get('width', 1)
        position = line_config.get('position', 'top right')
        target_val, target_col = None, y_col if axis == 'y' else x_col

        if target_col not in df.columns:
             st.warning(f"Reference line calculation skipped: Column '{target_col}' not found.")
             continue

        if isinstance(value, str) and value.lower() in ['mean', 'median']:
            # Ensure target column is numeric before calculating stats
            numeric_data = pd.to_numeric(df[target_col], errors='coerce').dropna()
            if not numeric_data.empty:
                target_val = numeric_data.mean() if value.lower() == 'mean' else numeric_data.median()
                label = f"{value.capitalize()} ({target_val:.3f})" if not label else f"{label} ({target_val:.3f})"
            else:
                 st.warning(f"Could not calculate {value} for reference line: Column '{target_col}' has no numeric data.")
                 continue
        else:
            try: target_val = float(value)
            except (ValueError, TypeError):
                st.warning(f"Invalid value '{value}' for reference line '{label}'. Must be numeric or 'mean'/'median'.")
                continue
        if target_val is not None:
            line_params = dict(line=dict(color=color, width=width, dash=style))
            annot_params = dict(annotation_text=label, annotation_position=position)
            try:
                if axis == 'y': fig.add_hline(y=target_val, **line_params, **annot_params)
                elif axis == 'x': fig.add_vline(x=target_val, **line_params, **annot_params)
            except Exception as e:
                 st.warning(f"Could not add reference line '{label}' at {target_val}: {e}")
    return fig

def process_dataset(df):
    """Process the dataset (original logic)."""
    df = df.copy()
    if 'evaluation_dataset' in df.columns:
        # st.info("Processing 'evaluation_dataset' into 'set_type' and 'set_number'.") # Reduce verbosity
        df['set_type'] = df['evaluation_dataset'].apply(extract_set_type)
        df['set_number'] = df['evaluation_dataset'].apply(get_set_number)

    # Define expected metric columns more carefully
    metric_cols_to_convert = ['label_accuracy', 'combined_accuracy', 'malformed_domain', 'malformed_label', 'correct_predictions']
    metric_cols_to_convert += [col for col in df.columns if '_accuracy' in col]
    metric_cols_to_convert += [col for col in df.columns if '_confidence' in col] # Add confidence cols
    metric_cols_to_convert = list(set(metric_cols_to_convert)) # Ensure unique

    # st.info(f"Attempting to convert columns to numeric: {metric_cols_to_convert}") # Reduce verbosity
    conversion_warnings = []
    for col in metric_cols_to_convert:
        if col in df.columns:
            original_dtype = df[col].dtype
            # Only convert if not already numeric
            if not pd.api.types.is_numeric_dtype(original_dtype):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().all():
                     conversion_warnings.append(f"Column '{col}' could not be converted to numeric and is now all NaN.")
                elif df[col].isnull().any():
                     conversion_warnings.append(f"Some non-numeric values in '{col}' converted to NaN.")
    # Display warnings once after loop
    # for warning in conversion_warnings: st.info(warning) # Reduce verbosity

    df = df.replace([np.inf, -np.inf], np.nan)
    nan_count = df.isna().sum()
    if nan_count.sum() > 0:
        st.sidebar.caption("NaN values found after processing:")
        st.sidebar.dataframe(nan_count[nan_count > 0])
    return df

# --- Sidebar Setup ---
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("Configure data loading, filtering, and global plot settings.")
st.sidebar.divider()

# Data Loading
st.sidebar.subheader("1. Load Data")
file_path_option = st.sidebar.radio("Data Source", ["Upload CSV", "Use Hardcoded Path"], horizontal=True, help="Choose how to load the results CSV data.")
df = None
# Use session state to store raw and processed dataframes
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'df_processed' not in st.session_state: st.session_state.df_processed = None
if 'loaded_file_name' not in st.session_state: st.session_state.loaded_file_name = None

if file_path_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your results CSV file", type=["csv"])
    if uploaded_file is not None:
        # Only reload if file is different
        if st.session_state.loaded_file_name != uploaded_file.name:
            try:
                df_raw = pd.read_csv(uploaded_file)
                st.session_state.raw_df = df_raw
                st.session_state.df_processed = process_dataset(df_raw.copy())
                st.session_state.loaded_file_name = uploaded_file.name
                # Clear dependent session state on new load
                st.session_state.pop('df_filtered', None)
                keys_to_clear = [k for k in st.session_state if k.startswith('lc_results_')]
                for k in keys_to_clear: del st.session_state[k]
                st.sidebar.success(f"Uploaded '{uploaded_file.name}' ({len(df_raw)} rows)")
                st.rerun() # Rerun to update filters based on new data
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
                st.session_state.raw_df = None; st.session_state.df_processed = None; st.session_state.loaded_file_name = None
                st.stop()
    # If no file uploaded but data exists in state, use it
    elif st.session_state.df_processed is None:
        st.info("Please upload a CSV file to begin analysis.")
        st.stop()

else: # Use Hardcoded Path
    file_path = st.sidebar.text_input("CSV File Path", value="final_run_results.csv") # Use the uploaded CSV name as default
    # Only reload if path changes or no data loaded yet
    if st.session_state.loaded_file_name != file_path or st.session_state.df_processed is None:
        try:
            df_raw = pd.read_csv(file_path)
            st.session_state.raw_df = df_raw
            st.session_state.df_processed = process_dataset(df_raw.copy())
            st.session_state.loaded_file_name = file_path
            # Clear dependent session state on new load
            st.session_state.pop('df_filtered', None)
            keys_to_clear = [k for k in st.session_state if k.startswith('lc_results_')]
            for k in keys_to_clear: del st.session_state[k]
            st.sidebar.success(f"Loaded data from path ({len(df_raw)} rows)")
            st.rerun() # Rerun to update filters based on new data
        except FileNotFoundError:
             st.sidebar.error(f"Error: File not found at path '{file_path}'. Please check the path or upload the file.")
             st.session_state.raw_df = None; st.session_state.df_processed = None; st.session_state.loaded_file_name = None
             st.stop()
        except Exception as e:
            st.sidebar.error(f"Error loading file from path: {str(e)}")
            st.session_state.raw_df = None; st.session_state.df_processed = None; st.session_state.loaded_file_name = None
            st.stop()

# Get the processed dataframe for filtering
df_processed = st.session_state.get('df_processed')
if df_processed is None:
    st.error("Data could not be loaded or processed.")
    st.stop()

# Deduplication Info
st.sidebar.caption(f"Original rows: {len(st.session_state.raw_df)}")
df_deduplicated = df_processed.drop_duplicates(keep='first')
num_duplicates = len(df_processed) - len(df_deduplicated)
if num_duplicates > 0: st.sidebar.info(f"Removed {num_duplicates} duplicate rows.")
df_processed = df_deduplicated # Use deduplicated data going forward
st.sidebar.caption(f"Rows after deduplication: {len(df_processed)}")

# Filtering
st.sidebar.divider()
st.sidebar.subheader("2. Filter Data")
st.sidebar.markdown("Select values to include in the analysis.")

# Initialize filtered df if it doesn't exist or if data changed
if 'df_filtered' not in st.session_state or st.session_state.df_filtered is None:
    st.session_state.df_filtered = df_processed.copy()

# Get columns suitable for filtering (exclude high cardinality, maybe numeric too unless specified)
potential_filter_cols = [col for col in df_processed.columns if df_processed[col].nunique() < 50 and not pd.api.types.is_numeric_dtype(df_processed[col]) or col == 'set_number']
filter_cols_selected = st.sidebar.multiselect(
    "Filter by columns:", options=potential_filter_cols,
    default=['set_type'] if 'set_type' in potential_filter_cols else [],
    help="Select categorical columns to apply filters on."
)

active_filters = {}
df_for_filtering = df_processed.copy() # Start fresh for applying filters

for col in filter_cols_selected:
    if col in df_for_filtering.columns:
        unique_vals = sorted(df_for_filtering[col].dropna().unique())
        if len(unique_vals) > 1:
            # Use session state to remember selections across reruns for filters
            filter_key = f"filter_select_{col}"
            if filter_key not in st.session_state:
                 st.session_state[filter_key] = unique_vals # Default to all selected initially
            selected_vals = st.sidebar.multiselect(f"'{clean_label(col)}' values:", options=unique_vals, key=filter_key)
            active_filters[col] = selected_vals

# Specific filter for set_number if available
if 'set_number' in df_processed.columns:
     unique_set_numbers = sorted(df_processed['set_number'].dropna().astype(int).unique()) # Ensure int for sorting
     filter_key_setnum = "filter_select_set_number"
     if filter_key_setnum not in st.session_state:
         st.session_state[filter_key_setnum] = unique_set_numbers
     selected_set_numbers = st.sidebar.multiselect("Filter by Set Number:", options=unique_set_numbers, key=filter_key_setnum)
     active_filters['set_number'] = selected_set_numbers

# Apply filters sequentially
df_currently_filtered = df_for_filtering
for col, selected_values in active_filters.items():
    if col in df_currently_filtered.columns and selected_values: # Only filter if values are selected
        df_currently_filtered = df_currently_filtered[df_currently_filtered[col].isin(selected_values)]

st.session_state.df_filtered = df_currently_filtered # Store the final filtered df

if st.session_state.df_filtered.empty:
    st.sidebar.error("No data matches the selected filters!")
    st.warning("Current filters result in an empty dataset. Please adjust filters in the sidebar.")
else:
    st.sidebar.success(f"Filtered data: {len(st.session_state.df_filtered)} rows")

# Global Plot Settings
st.sidebar.divider()
st.sidebar.subheader("3. Global Plot Settings")
color_palette = st.sidebar.selectbox(
    "Color Palette", options=['Plotly', 'Set1', 'Set2', 'Set3','Alphabet','D3', 'Bold', 'Prism', 'Dark24', 'Light24'], index=0, # Default to Plotly
    key="global_color_palette", help="Select the default color scheme for plots."
)
try: colors = getattr(px.colors.qualitative, color_palette)
except AttributeError: st.sidebar.warning(f"Palette '{color_palette}' not found, using Plotly."); colors = px.colors.qualitative.Plotly
pio.templates.default = "plotly_white"

# --- Main App Area ---
st.title('📊 Model Performance Analysis Dashboard')
st.markdown("Explore and visualize the results from the model evaluation experiments documented in the paper.")
st.caption(f"Currently analyzing **{len(st.session_state.df_filtered)}** rows after filtering.")
st.divider()

if st.session_state.df_filtered is None or st.session_state.df_filtered.empty:
    st.warning("No data available for analysis based on current filters. Adjust filters in the sidebar or load data.")
    st.stop()

df_filtered = st.session_state.df_filtered

tab1, tab2 = st.tabs(["📊 Metrics Analysis", "📈 Learning Curve Analysis"])

# ============================ Metrics Analysis Tab ============================
with tab1:
    st.header('Metrics Distribution Analysis')
    st.markdown("Use box plots to compare the distribution of performance metrics across different experimental settings (e.g., model, prompt type).")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Select Metrics & Groupings")
        metric_cols = ['label_accuracy', 'combined_accuracy', 'malformed_domain', 'malformed_label', 'correct_predictions']
        metric_cols += [col for col in df_filtered.columns if '_accuracy' in col]
        metric_cols += [col for col in df_filtered.columns if '_confidence' in col]
        metric_cols = sorted(list(set([col for col in metric_cols if col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col])]))) # Ensure numeric
        if not metric_cols: metric_cols = get_numeric_columns(df_filtered)

        # Use session state for multiselect default
        if 'dist_metrics_selected' not in st.session_state:
             st.session_state.dist_metrics_selected = [metric_cols[0]] if metric_cols else []
        metrics = st.multiselect("Select Metric(s) to Plot", metric_cols, key="dist_metrics_select_widget", default=st.session_state.dist_metrics_selected, help="Choose one or more metrics for the Y-axis.")
        st.session_state.dist_metrics_selected = metrics # Update session state

    with col2:
        st.subheader("Define Plot Structure")
        valid_grouping_cols = [col for col in df_filtered.columns if df_filtered[col].nunique() < 50 and col not in metrics]
        suggested_groupings = ['model_name', 'response_type', 'response_format', 'explanation_type', 'generation_strategy', 'prompt_type', 'num_training_samples', 'set_type', 'set_number']
        default_groupings = [col for col in suggested_groupings if col in valid_grouping_cols]
        # Use session state for multiselect default
        if 'dist_groupings_selected' not in st.session_state:
            st.session_state.dist_groupings_selected = default_groupings[:4] if default_groupings else []
        available_groupings = st.multiselect("Available Grouping Columns", valid_grouping_cols, key="dist_groupings_select_widget", default=st.session_state.dist_groupings_selected, help="Select columns defining experimental conditions.")
        st.session_state.dist_groupings_selected = available_groupings

    if available_groupings:
        st.markdown("##### Assign Grouping Roles")
        col1, col2 = st.columns(2)
        with col1:
            primary_grouping = st.selectbox("Primary Grouping (X-axis)", options=["None"] + available_groupings, index=available_groupings.index('model_name') + 1 if 'model_name' in available_groupings else (1 if available_groupings else 0), help="Main category for the X-axis.")
            remaining_for_tertiary = ["None"] + [g for g in available_groupings if g != primary_grouping]
            tertiary_grouping = st.selectbox("Tertiary Grouping (Facet Columns)", options=remaining_for_tertiary, index=0, help="Create separate plot columns for each value.")
        with col2:
            remaining_for_secondary = ["None"] + [g for g in available_groupings if g != primary_grouping]
            secondary_grouping = st.selectbox("Secondary Grouping (Color)", options=remaining_for_secondary, index=1 if len(remaining_for_secondary) > 1 else 0, help="Color boxes based on this category.")
            remaining_for_quaternary = ["None"] + [g for g in available_groupings if g != primary_grouping and g != secondary_grouping and g != tertiary_grouping]
            quaternary_grouping = st.selectbox("Quaternary Grouping (Facet Rows)", options=remaining_for_quaternary, index=0, help="Create separate plot rows for each value.")

        st.markdown("##### Plot Options")
        col1, col2 = st.columns(2)
        with col1: show_sample_counts = st.checkbox("Show sample counts (n=...)", value=False, key="dist_show_counts", help="Display the number of data points for each box.")
        with col2: sort_x_axis = st.radio("Sort X-axis By:", ["No Sorting", "Ascending", "Descending"], index=0, horizontal=True, key="dist_sort")

        if metrics and primary_grouping != "None":
            st.divider(); st.subheader("📊 Visualization")
            df_plot = df_filtered.copy()
            if sort_x_axis != "No Sorting": df_plot = sort_data_by_column(df_plot, primary_grouping, sort_x_axis)

            facet_col = None if tertiary_grouping == "None" else tertiary_grouping
            facet_row = None if quaternary_grouping == "None" else quaternary_grouping
            color_col = None if secondary_grouping == "None" else secondary_grouping

            y_plot_col = metrics[0]; color_plot_col = color_col; facet_col_plot_col = facet_col; facet_row_plot_col = facet_row; plot_df = df_plot

            if len(metrics) > 1:
                st.info("Multiple metrics selected. Choose display mode:")
                display_mode = st.radio("Metrics Display Mode", ["Side-by-Side (Color)", "Separate Facets (Row)"], horizontal=True, index=0, key="multi_metric_mode")
                id_vars = [col for col in [primary_grouping, color_col, facet_col, facet_row] if col]
                other_cols = [c for c in df_plot.columns if c not in metrics and c not in id_vars]
                try:
                     melted_df = pd.melt(df_plot, id_vars=id_vars + other_cols, value_vars=metrics, var_name='_metric_variable_', value_name='_metric_value_')
                     melted_df['_metric_value_'] = pd.to_numeric(melted_df['_metric_value_'], errors='coerce')
                     plot_df = melted_df; y_plot_col = '_metric_value_'
                     if display_mode == "Side-by-Side (Color)":
                         if color_col: st.warning(f"Color grouping by '{color_col}' overridden by metrics.")
                         color_plot_col = '_metric_variable_'; facet_row_plot_col = facet_row
                     else:
                         if facet_row: st.warning(f"Facet Row by '{facet_row}' overridden by metrics.")
                         color_plot_col = color_col; facet_row_plot_col = '_metric_variable_'
                     facet_col_plot_col = facet_col
                except Exception as melt_error: st.error(f"Error preparing data for multi-metric plot: {melt_error}"); plot_df = None

            if plot_df is not None:
                plot_args = {"data_frame": plot_df, "x": primary_grouping, "y": y_plot_col, "color_discrete_sequence": colors}
                if color_plot_col: plot_args["color"] = color_plot_col
                if facet_col_plot_col: plot_args["facet_col"] = facet_col_plot_col
                if facet_row_plot_col: plot_args["facet_row"] = facet_row_plot_col
                if sort_x_axis != "No Sorting" and primary_grouping in plot_df.columns:
                    unique_sorted_cats = list(plot_df[primary_grouping].dropna().unique())
                    if sort_x_axis == "Ascending": unique_sorted_cats.sort()
                    elif sort_x_axis == "Descending": unique_sorted_cats.sort(reverse=True)
                    plot_args["category_orders"] = {primary_grouping: unique_sorted_cats}

                try:
                    fig = px.box(**plot_args)
                    metrics_to_ref = metrics if len(metrics) == 1 else plot_df['_metric_variable_'].unique() if '_metric_variable_' in plot_df else []
                    if any(m in ['label_accuracy', 'combined_accuracy'] for m in metrics_to_ref):
                        fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Random Chance (0.25)", annotation_position="bottom right")
                        fig.add_hline(y=0.675, line_dash="dash", line_color="purple", annotation_text="Human baseline (0.675)", annotation_position="bottom right")

                    create_and_update_plot(fig, plot_df, primary_grouping, y_plot_col, color_plot_col, facet_row_plot_col, facet_col_plot_col, 'box', "dist_plot", colors)

                    with st.expander("View Summary Statistics"):
                        st.markdown(f"Statistics for **{', '.join(map(clean_label, metrics))}**, grouped by plot dimensions.")
                        group_cols_stats = [col for col in [primary_grouping, color_plot_col, facet_col_plot_col, facet_row_plot_col] if col]
                        valid_group_cols_stats = [col for col in group_cols_stats if col in plot_df.columns]
                        if valid_group_cols_stats:
                            try:
                                summary = plot_df.groupby(valid_group_cols_stats)[y_plot_col].agg(['count', 'mean', 'std', 'median', 'min', 'max']).round(4)
                                st.dataframe(summary)
                            except Exception as e: st.warning(f"Could not calculate grouped stats: {e}")
                        else: st.dataframe(plot_df[[y_plot_col]].describe().round(4))

                except Exception as e:
                    st.error(f"Error generating box plot: {e}")
                    st.code(traceback.format_exc())
        else:
             if not metrics: st.warning("Please select at least one metric.")
             if primary_grouping == "None": st.warning("Please select a Primary Grouping (X-axis).")
    else: st.warning("Please select available grouping columns.")
    st.divider()

# ============================ Learning Curve Tab ============================
with tab2:
    st.header('Learning Curve Analysis')
    st.markdown("Analyze how `label_accuracy` changes with `num_training_samples`, fitting a power law curve (`y = a - b*x^-c`) to the median accuracy. This helps understand how model performance scales with data.")
    st.divider()

    if df_filtered is not None and not df_filtered.empty:
        required_cols = ['num_training_samples', 'label_accuracy', 'model_name']
        if not all(col in df_filtered.columns for col in required_cols):
            st.warning(f"Required columns missing for this analysis: Need `{'`, `'.join(required_cols)}`.")
            st.stop()

        st.subheader("Select Groups and Options")

        # --- MODIFICATION: Add Composite Grouping UI ---
        # Store previous state to detect changes
        prev_use_composite = st.session_state.get("lc_composite_check_prev", False)
        use_composite_groups = st.checkbox(
            "Use composite grouping (combine multiple parameters)",
            value=prev_use_composite, # Use previous state as default
            key="lc_composite_check",
            help="Group curves by combinations of parameters (e.g., Model + Prompt Type) instead of just Model Name."
        )
        st.session_state.lc_composite_check_prev = use_composite_groups # Store current state

        group_col_lc = 'model_name' # Default grouping column
        df_lc_analysis = df_filtered.copy() # Start with filtered data
        group_params = [] # Initialize group_params

        if use_composite_groups:
            st.markdown("##### Define Composite Group")
            potential_group_params = [col for col in df_lc_analysis.columns if
                               col not in ['num_training_samples', 'label_accuracy'] and
                               df_lc_analysis[col].nunique() < 50]
            # Store previous selection to detect changes
            prev_group_params = st.session_state.get("lc_composite_params_prev", ['model_name'] if 'model_name' in potential_group_params else [])
            if 'lc_composite_params_selected' not in st.session_state:
                 st.session_state.lc_composite_params_selected = prev_group_params

            group_params = st.multiselect(
                "Select parameters to combine for grouping:",
                options=potential_group_params,
                key="lc_composite_params_widget",
                default=st.session_state.lc_composite_params_selected,
                help="Select two or more columns. The unique combinations will form the groups."
            )
            st.session_state.lc_composite_params_selected = group_params # Update state

            if len(group_params) >= 1:
                try:
                    df_lc_analysis['composite_group'] = df_lc_analysis[group_params].astype(str).agg(' | '.join, axis=1)
                    group_col_lc = 'composite_group'
                    st.caption(f"Grouping by combinations of: {', '.join(map(clean_label, group_params))}")
                except KeyError as e: st.error(f"Error creating composite group: Column '{e}' not found."); st.stop()
                except Exception as e: st.error(f"Error creating composite group: {e}"); st.stop()
            else:
                st.warning("Please select at least one parameter to create composite groups.")
                group_col_lc = 'model_name' # Fallback

        # --- Check if grouping configuration changed ---
        grouping_config_changed = False
        current_group_col = group_col_lc
        current_composite_params = tuple(sorted(group_params)) if use_composite_groups else None

        if st.session_state.get("lc_last_group_col") != current_group_col:
            grouping_config_changed = True
        if use_composite_groups and st.session_state.get("lc_last_composite_params") != current_composite_params:
            grouping_config_changed = True

        # Store current config for next run comparison
        st.session_state["lc_last_group_col"] = current_group_col
        if use_composite_groups:
            st.session_state["lc_last_composite_params"] = current_composite_params
        else:
            st.session_state.pop("lc_last_composite_params", None) # Remove if not used

        # --- Group Selection Multiselect ---
        col1, col2 = st.columns(2)
        with col1:
            all_groups = sorted(df_lc_analysis[group_col_lc].unique())
            default_groups_key = f"lc_selected_groups_{group_col_lc}" # Key depends on grouping column

            # Reset default if grouping config changed or key doesn't exist
            if grouping_config_changed or default_groups_key not in st.session_state:
                st.session_state[default_groups_key] = all_groups[:min(5, len(all_groups))]

            selected_groups = st.multiselect(f"Select {clean_label(group_col_lc)} to Analyze", options=all_groups, default=st.session_state[default_groups_key], help="Choose which groups to include.", key="lc_select_groups_widget")
            # Update session state with current selection (important!)
            st.session_state[default_groups_key] = selected_groups

            # Filter based on selected groups
            df_lc = df_lc_analysis[df_lc_analysis[group_col_lc].isin(selected_groups)].copy()

        with col2:
            log_x_axis = st.checkbox("Use Log Scale for X-axis", value=True, help="Display the number of training samples on a logarithmic scale.")
            show_error_bars = st.checkbox("Show IQR Error Bars", value=True, help="Show the interquartile range (25th to 75th percentile).")

        run_lc_analysis = st.button("📈 Generate Learning Curve Analysis", key="run_lc_analysis")
        st.divider()

        # Use session state to store results, key includes group column name AND selected groups
        session_key_lc = f"lc_results_{group_col_lc}_{'_'.join(sorted(map(str, selected_groups)))}" # Ensure key is robust

        if run_lc_analysis:
            if df_lc.empty: st.error("No data selected for Learning Curve analysis.")
            else:
                with st.spinner("Analyzing learning curves..."):
                    # Pass the correct group_col to analysis functions
                    grouped_lc, fit_params_lc, r_squared_lc = analyze_learning_curve(df_lc, group_col=group_col_lc)
                    metrics_lc = calculate_efficiency_metrics(grouped_lc, fit_params_lc, group_col=group_col_lc)
                    st.session_state[session_key_lc] = {'grouped': grouped_lc, 'fit_params': fit_params_lc, 'r_squared': r_squared_lc, 'metrics': metrics_lc, 'data': df_lc, 'group_col': group_col_lc} # Store group_col used

        if session_key_lc in st.session_state and st.session_state[session_key_lc]:
            results = st.session_state[session_key_lc]
            grouped_lc, fit_params_lc, r_squared_lc, metrics_lc, df_lc_analyzed, used_group_col = results['grouped'], results['fit_params'], results['r_squared'], results['metrics'], results['data'], results['group_col']

            if not grouped_lc.empty:
                st.subheader("📈 Learning Curve Visualization")
                fig_lc = plot_learning_curves_plotly(grouped_lc, fit_params_lc, r_squared_lc, use_log_x=log_x_axis, show_error_bars=show_error_bars, colors=colors, group_col=used_group_col)
                # Pass df_lc_analyzed which is the data used for this specific run
                create_and_update_plot(fig_lc, df_lc_analyzed, 'num_training_samples', 'label_accuracy', used_group_col, None, None, 'scatter', "lc_plot", colors)

                st.divider(); st.subheader("🚀 Learning Efficiency Metrics")
                st.markdown("Quantifies aspects of the learning process based on the median accuracy curve.")
                if metrics_lc:
                    metrics_df = pd.DataFrame.from_dict(metrics_lc, orient='index')
                    if r_squared_lc: metrics_df['R_squared'] = metrics_df.index.map(r_squared_lc).fillna(np.nan)
                    display_cols = ['baseline_accuracy', 'max_accuracy', 'total_improvement', 'relative_improvement', 'sample_efficiency', 'plateau_samples', 'plateau_accuracy', 'R_squared']
                    metrics_display = metrics_df[[col for col in display_cols if col in metrics_df.columns]].copy()
                    metrics_display.columns = [clean_label(col) for col in metrics_display.columns]
                    st.dataframe(metrics_display.round(4))
                    csv_metrics_lc = metrics_df.to_csv(index=True).encode('utf-8')
                    st.download_button("Download Efficiency Metrics CSV", csv_metrics_lc, "learning_curve_metrics.csv", "text/csv", key="dl_lc_metrics")
                else: st.info("Could not calculate efficiency metrics.")

                st.divider(); st.subheader("🌡️ Percentage of Maximum Gain Heatmap")
                st.markdown("Shows how quickly each group reaches its potential maximum accuracy gain.")
                if metrics_lc:
                    metrics_df_for_heatmap = pd.DataFrame.from_dict(metrics_lc, orient='index')
                    pct_gain_cols = [col for col in metrics_df_for_heatmap.columns if 'pct_gain_at_' in col]
                    if pct_gain_cols:
                        pct_gain_df = metrics_df_for_heatmap[pct_gain_cols].copy().fillna(0)
                        try:
                             pct_gain_df.columns = [int(col.split('_')[-1]) for col in pct_gain_cols]
                             pct_gain_df = pct_gain_df.reindex(sorted(pct_gain_df.columns), axis=1)
                        except ValueError: st.caption("Could not sort heatmap columns numerically.")
                        fig_heatmap = px.imshow(pct_gain_df.T, labels=dict(x=clean_label(used_group_col), y="Training Samples", color="% Max Gain"), color_continuous_scale="Viridis", aspect="auto")
                        fig_heatmap.update_layout(title="% Max Gain vs Training Samples", height=max(400, len(pct_gain_df.columns)*20))
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else: st.info("Not enough data points to generate gain heatmap.")
                else: st.info("Metrics needed for heatmap were not calculated.")

                st.divider(); st.subheader("⚙️ Curve Fit Parameters")
                if fit_params_lc:
                    fit_params_disp = {}
                    for group, params in fit_params_lc.items():
                        if params is not None: fit_params_disp[group] = {'a (Asymptote)': params[0], 'b (Scale)': params[1], 'c (Rate)': params[2], 'R_squared': r_squared_lc.get(group)}
                    if fit_params_disp:
                        fit_params_df = pd.DataFrame.from_dict(fit_params_disp, orient='index')
                        st.dataframe(fit_params_df.round(4))
                        with st.expander("Power Law Fit Explanation", expanded=True):
                            st.markdown("""**Understanding the Power Law Fit Parameters**: `Accuracy = a - b * (Samples ^ -c)`
                            - **`a` (Asymptote)**: Theoretical max accuracy as samples → ∞. Higher is better.
                            - **`b` (Scale)**: Influences the vertical span of the curve.
                            - **`c` (Rate)**: Learning rate exponent. Higher = faster initial learning.
                            - **`R_squared`**: Goodness of fit (1.0 = perfect). Closer to 1 suggests power law is a good model here.""")
                        csv_fit_lc = fit_params_df.to_csv(index=True).encode('utf-8')
                        st.download_button("Download Fit Parameters CSV", csv_fit_lc, "learning_curve_fit_params.csv", "text/csv", key="dl_lc_fit_params")
                    else: st.info("No successful curve fits to display parameters for.")
                else: st.info("Curve fitting was not successful for any selected group.")
            else: st.warning("Analysis did not yield results. Check data and filters.")
        elif run_lc_analysis: st.warning("Analysis could not be completed. Check data and filters.")

    else: st.warning("Please load data using the sidebar to enable Learning Curve Analysis.")
    st.divider()

# --- Display Raw Data ---
if df_filtered is not None:
    st.divider()
    with st.expander("View Filtered Data Sample"):
        st.markdown(f"Showing the first 100 rows of the **{len(df_filtered)}** currently filtered data points.")
        st.dataframe(df_filtered.head(100))
