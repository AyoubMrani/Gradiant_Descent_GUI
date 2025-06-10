import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

isNormalized = False

class DataAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis & Gradient Descent")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Data variables
        self.data = None
        self.processed_data = None
        self.columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.boolean_columns = []

        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.data_info_tab = ttk.Frame(self.notebook)
        self.missing_tab = ttk.Frame(self.notebook)
        self.preprocessing_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.data_tab, text="Data Loading")
        self.notebook.add(self.data_info_tab, text="Data Information")
        self.notebook.add(self.missing_tab, text="Missing Values")
        self.notebook.add(self.preprocessing_tab, text="Preprocessing")
        self.notebook.add(self.visualization_tab, text="Visualization")
        self.notebook.add(self.model_tab, text="Prediction")

        # Setup tabs
        self.setup_data_tab()
        self.setup_data_info_tab()
        self.setup_missing_tab()
        self.setup_preprocessing_tab()
        self.setup_visualization_tab()
        self.setup_model_tab()

    
    def setup_data_tab(self):
        # File loading section
        file_frame = ttk.LabelFrame(self.data_tab, text="Load Dataset", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(file_frame, text="Browse CSV File", command=self.load_csv).pack(pady=10)
        
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(pady=5)
        
        # Data preview section
        preview_frame = ttk.LabelFrame(self.data_tab, text="Data Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview for data preview
        self.tree_frame = ttk.Frame(preview_frame)
        self.tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree_scroll_y = ttk.Scrollbar(self.tree_frame)
        self.tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree_scroll_x = ttk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL)
        self.tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tree = ttk.Treeview(self.tree_frame, yscrollcommand=self.tree_scroll_y.set, 
                                xscrollcommand=self.tree_scroll_x.set)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        self.tree_scroll_y.config(command=self.tree.yview)
        self.tree_scroll_x.config(command=self.tree.xview)
        
        # Data info section
        info_frame = ttk.LabelFrame(self.data_tab, text="Data Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        

    
    def setup_missing_tab(self):
        # Main horizontal split frame
        main_frame = ttk.Frame(self.missing_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for summary and by column
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Right frame for handling and visualization
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # --- Left: Missing values summary ---
        summary_frame = ttk.LabelFrame(left_frame, text="Missing Values Summary", padding=10)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 2))

        summary_container = ttk.Frame(summary_frame)
        summary_container.pack(fill=tk.BOTH, expand=True)
        self.missing_summary_text = tk.Text(summary_container, height=10, wrap=tk.WORD)
        self.missing_summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll = ttk.Scrollbar(summary_container, command=self.missing_summary_text.yview)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.missing_summary_text.config(yscrollcommand=summary_scroll.set)
        self.missing_summary_text.config(state=tk.DISABLED)

        # --- Left: Missing values by column ---
        columns_frame = ttk.LabelFrame(left_frame, text="Missing Values by Column", padding=10)
        columns_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))

        columns_container = ttk.Frame(columns_frame)
        columns_container.pack(fill=tk.BOTH, expand=True)
        self.missing_columns_text = tk.Text(columns_container, height=15, wrap=tk.WORD)
        self.missing_columns_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        columns_scroll = ttk.Scrollbar(columns_container, command=self.missing_columns_text.yview)
        columns_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.missing_columns_text.config(yscrollcommand=columns_scroll.set)
        self.missing_columns_text.config(state=tk.DISABLED)

        # --- Right: Handle missing values ---
        handling_frame = ttk.LabelFrame(right_frame, text="Handle Missing Values", padding=10)
        handling_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=(5, 2))

        strategy_frame = ttk.Frame(handling_frame)
        strategy_frame.pack(fill=tk.X, pady=10)
        ttk.Label(strategy_frame, text="Strategy:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.strategy_var = tk.StringVar(value="mean")
        strategy_combo = ttk.Combobox(strategy_frame, textvariable=self.strategy_var, 
                            values=["mean", "constant", "most frequent", "drop"], state="readonly")
        strategy_combo.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(strategy_frame, text="Fill Value (for constant):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.fill_value_var = tk.StringVar(value="0")
        ttk.Entry(strategy_frame, textvariable=self.fill_value_var).grid(row=1, column=1, padx=5, pady=5)

        columns_selection_frame = ttk.Frame(handling_frame)
        columns_selection_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        ttk.Label(columns_selection_frame, text="Select columns to handle:").pack(anchor=tk.W)
        self.columns_canvas = tk.Canvas(columns_selection_frame)
        scrollbar = ttk.Scrollbar(columns_selection_frame, orient=tk.VERTICAL, command=self.columns_canvas.yview)
        self.columns_scrollable_frame = ttk.Frame(self.columns_canvas)
        self.columns_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.columns_canvas.configure(scrollregion=self.columns_canvas.bbox("all"))
        )
        self.columns_canvas.create_window((0, 0), window=self.columns_scrollable_frame, anchor=tk.NW)
        self.columns_canvas.configure(yscrollcommand=scrollbar.set)
        self.columns_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Button(handling_frame, text="Apply", command=self.handle_missing_values).pack(pady=10)

        
    #ayoub
    def setup_visualization_tab(self):
        # Create notebook for visualization options
        self.viz_notebook = ttk.Notebook(self.visualization_tab)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create visualization sub-tabs
        self.target_viz_tab = ttk.Frame(self.viz_notebook)
        self.corr_viz_tab = ttk.Frame(self.viz_notebook)
        # self.dist_viz_tab = ttk.Frame(self.viz_notebook)
        
        self.viz_notebook.add(self.target_viz_tab, text="Histogram Distribution")
        self.viz_notebook.add(self.corr_viz_tab, text="Correlation Analysis")
        # self.viz_notebook.add(self.dist_viz_tab, text="Feature Distributions")
        
        # Setup target visualization tab
        self.setup_target_viz_tab()
        self.setup_corr_viz_tab()
        # self.setup_dist_viz_tab()
    
    def setup_target_viz_tab(self):
        # Target selection
        selection_frame = ttk.Frame(self.target_viz_tab)
        selection_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(selection_frame, text="Select Target Variable:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.target_viz_var = tk.StringVar()
        self.target_viz_dropdown = ttk.Combobox(selection_frame, textvariable=self.target_viz_var, state="readonly")
        self.target_viz_dropdown.grid(row=0, column=1, padx=5, pady=5)
        
        self.target_plot_type = tk.StringVar(value="histogram")  # Only used internally
        
        # Plot button
        ttk.Button(selection_frame, text="Generate Plot", command=self.plot_target).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Plot frame
        self.target_plot_frame = ttk.Frame(self.target_viz_tab)
        self.target_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_corr_viz_tab(self):
        # Controls frame
        controls_frame = ttk.Frame(self.corr_viz_tab)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Target selection for correlation
        ttk.Label(controls_frame, text="Select Target for Correlation:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.corr_target_var = tk.StringVar()
        self.corr_target_dropdown = ttk.Combobox(controls_frame, textvariable=self.corr_target_var, state="readonly")
        self.corr_target_dropdown.grid(row=0, column=1, padx=5, pady=5)
        
        # Plot type
        ttk.Label(controls_frame, text="Plot Type:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.corr_plot_type = tk.StringVar(value="heatmap")
        corr_plot_combo = ttk.Combobox(controls_frame, textvariable=self.corr_plot_type, 
                                      values=["heatmap", "bar"], state="readonly")
        corr_plot_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Plot button
        ttk.Button(controls_frame, text="Generate Correlation Plot", 
                  command=self.plot_correlation).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Plot frame
        self.corr_plot_frame = ttk.Frame(self.corr_viz_tab)
        self.corr_plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    
    def setup_preprocessing_tab(self):
        # Create notebook for preprocessing options
        self.preprocessing_notebook = ttk.Notebook(self.preprocessing_tab)
        self.preprocessing_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create preprocessing sub-tabs
        self.dummies_tab = ttk.Frame(self.preprocessing_notebook)
        self.mapping_tab = ttk.Frame(self.preprocessing_notebook)
        self.normalize_tab = ttk.Frame(self.preprocessing_notebook)
        self.boolean_tab = ttk.Frame(self.preprocessing_notebook)
        
        self.preprocessing_notebook.add(self.dummies_tab, text="Dummy Variables")
        self.preprocessing_notebook.add(self.mapping_tab, text="Yes/No Mapping")
        self.preprocessing_notebook.add(self.normalize_tab, text="Normalization")
        self.preprocessing_notebook.add(self.boolean_tab, text="Boolean Conversion")
        
        # Setup dummy variables tab
        self.setup_dummies_tab()
        self.setup_mapping_tab()
        self.setup_normalize_tab()
        self.setup_boolean_tab()
    
    def setup_dummies_tab(self):
        ttk.Label(self.dummies_tab, text="Create dummy variables from categorical columns").pack(pady=10)
        
        # Drop first option
        self.drop_first_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.dummies_tab, text="Drop first category (recommended to avoid multicollinearity)", 
                       variable=self.drop_first_var).pack(pady=5, anchor=tk.W)
        
        # Categorical columns frame
        columns_frame = ttk.LabelFrame(self.dummies_tab, text="Select categorical columns")
        columns_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollable frame for columns
        self.dummies_canvas = tk.Canvas(columns_frame)
        scrollbar = ttk.Scrollbar(columns_frame, orient=tk.VERTICAL, command=self.dummies_canvas.yview)
        self.dummies_scrollable_frame = ttk.Frame(self.dummies_canvas)
        
        self.dummies_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.dummies_canvas.configure(scrollregion=self.dummies_canvas.bbox("all"))
        )
        
        self.dummies_canvas.create_window((0, 0), window=self.dummies_scrollable_frame, anchor=tk.NW)
        self.dummies_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.dummies_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Apply button
        ttk.Button(self.dummies_tab, text="Apply Get Dummies", command=self.apply_get_dummies).pack(pady=10)
    
    def setup_mapping_tab(self):
        ttk.Label(self.mapping_tab, text="Map categorical values (e.g., 'yes'/'no') to 1/0").pack(pady=10)
        
        # Mapping values
        mapping_values_frame = ttk.Frame(self.mapping_tab)
        mapping_values_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(mapping_values_frame, text="Value for 'Yes':").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.yes_value = tk.StringVar(value="yes")
        ttk.Entry(mapping_values_frame, textvariable=self.yes_value).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(mapping_values_frame, text="Value for 'No':").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.no_value = tk.StringVar(value="no")
        ttk.Entry(mapping_values_frame, textvariable=self.no_value).grid(row=1, column=1, padx=5, pady=5)
        
        # Categorical columns frame
        columns_frame = ttk.LabelFrame(self.mapping_tab, text="Select columns to map")
        columns_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollable frame for columns
        self.mapping_canvas = tk.Canvas(columns_frame)
        scrollbar = ttk.Scrollbar(columns_frame, orient=tk.VERTICAL, command=self.mapping_canvas.yview)
        self.mapping_scrollable_frame = ttk.Frame(self.mapping_canvas)
        
        self.mapping_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.mapping_canvas.configure(scrollregion=self.mapping_canvas.bbox("all"))
        )
        
        self.mapping_canvas.create_window((0, 0), window=self.mapping_scrollable_frame, anchor=tk.NW)
        self.mapping_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.mapping_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Apply button
        ttk.Button(self.mapping_tab, text="Apply Mapping", command=self.apply_mapping).pack(pady=10)
    
    def setup_normalize_tab(self):
        ttk.Label(self.normalize_tab, text="Normalize numerical features to range [0, 1]").pack(pady=10)
        
        # Numerical columns frame
        columns_frame = ttk.LabelFrame(self.normalize_tab, text="Select columns to normalize")
        columns_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollable frame for columns
        self.normalize_canvas = tk.Canvas(columns_frame)
        scrollbar = ttk.Scrollbar(columns_frame, orient=tk.VERTICAL, command=self.normalize_canvas.yview)
        self.normalize_scrollable_frame = ttk.Frame(self.normalize_canvas)
        
        self.normalize_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.normalize_canvas.configure(scrollregion=self.normalize_canvas.bbox("all"))
        )
        
        self.normalize_canvas.create_window((0, 0), window=self.normalize_scrollable_frame, anchor=tk.NW)
        self.normalize_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.normalize_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Apply button
        ttk.Button(self.normalize_tab, text="Apply Normalization", command=self.apply_normalization).pack(pady=10)
    
    def setup_boolean_tab(self):
        ttk.Label(self.boolean_tab, text="Convert boolean columns to integers (True=1, False=0)").pack(pady=10)
        
        # Boolean columns frame
        columns_frame = ttk.LabelFrame(self.boolean_tab, text="Select boolean columns to convert")
        columns_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollable frame for columns
        self.boolean_canvas = tk.Canvas(columns_frame)
        scrollbar = ttk.Scrollbar(columns_frame, orient=tk.VERTICAL, command=self.boolean_canvas.yview)
        self.boolean_scrollable_frame = ttk.Frame(self.boolean_canvas)
        
        self.boolean_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.boolean_canvas.configure(scrollregion=self.boolean_canvas.bbox("all"))
        )
        
        self.boolean_canvas.create_window((0, 0), window=self.boolean_scrollable_frame, anchor=tk.NW)
        self.boolean_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.boolean_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Apply button
        ttk.Button(self.boolean_tab, text="Convert to Integer", command=self.apply_boolean_conversion).pack(pady=10)
    
    def setup_model_tab(self):
        # Split into left and right frames
        model_frame = ttk.Frame(self.model_tab)
        model_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.Frame(model_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_frame = ttk.Frame(model_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Model configuration (left frame)
        config_frame = ttk.LabelFrame(left_frame, text="Model Configuration")
        # Prediction type selection
        ttk.Label(config_frame, text="Prediction Type:").pack(anchor=tk.W, padx=10, pady=5)
        self.prediction_type_var = tk.StringVar(value="Regression")
        prediction_type_combo = ttk.Combobox(
            config_frame,
            textvariable=self.prediction_type_var,
            values=["Regression", "Classification"],
            state="readonly"
        )
        prediction_type_combo.pack(fill=tk.X, padx=10, pady=5)

        config_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Target variable
        ttk.Label(config_frame, text="Target Variable:").pack(anchor=tk.W, padx=10, pady=5)
        self.target_var = tk.StringVar()
        self.target_dropdown = ttk.Combobox(config_frame, textvariable=self.target_var, state="readonly")
        self.target_dropdown.pack(fill=tk.X, padx=10, pady=5)
        
        # Features
        ttk.Label(config_frame, text="Features:").pack(anchor=tk.W, padx=10, pady=5)
        
        features_frame = ttk.Frame(config_frame)
        features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.features_canvas = tk.Canvas(features_frame)
        scrollbar = ttk.Scrollbar(features_frame, orient=tk.VERTICAL, command=self.features_canvas.yview)
        self.features_scrollable_frame = ttk.Frame(self.features_canvas)
        
        self.features_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.features_canvas.configure(scrollregion=self.features_canvas.bbox("all"))
        )
        
        self.features_canvas.create_window((0, 0), window=self.features_scrollable_frame, anchor=tk.NW)
        self.features_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.features_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add intercept
        self.add_intercept_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="Add intercept (recommended)", 
                       variable=self.add_intercept_var).pack(anchor=tk.W, padx=10, pady=5)
        
        # Model parameters
        params_frame = ttk.Frame(config_frame)
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(params_frame, text="Test Size:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(params_frame, from_=0.0, to=0.5, increment=0.05, 
                   textvariable=self.test_size_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Learning Rate (α):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.learning_rate_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(params_frame, from_=0.0001, to=0.1, increment=0.001, 
                   textvariable=self.learning_rate_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Iterations:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.iterations_var = tk.IntVar(value=1000)
        ttk.Spinbox(params_frame, from_=100, to=10000, increment=100, 
                   textvariable=self.iterations_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # Run button
        ttk.Button(config_frame, text="Run Prediction", command=self.run_gradient_descent).pack(pady=10)

        
        # Results frame (right frame)
        self.results_frame = ttk.LabelFrame(right_frame, text="Model Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        # Scrollable prediction frame
        self.predict_outer_frame = ttk.LabelFrame(right_frame, text="Make Prediction")
        self.predict_outer_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.predict_canvas = tk.Canvas(self.predict_outer_frame)
        self.predict_scrollbar = ttk.Scrollbar(self.predict_outer_frame, orient=tk.VERTICAL, command=self.predict_canvas.yview)
        self.predict_inner_frame = ttk.Frame(self.predict_canvas)

        self.predict_inner_frame.bind(
            "<Configure>",
            lambda e: self.predict_canvas.configure(scrollregion=self.predict_canvas.bbox("all"))
        )

        self.predict_canvas.create_window((0, 0), window=self.predict_inner_frame, anchor="nw")
        self.predict_canvas.configure(yscrollcommand=self.predict_scrollbar.set)

        self.predict_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.predict_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        
        # Initially hide results
        ttk.Label(self.results_frame, text="Run the model to see results").pack(pady=20)
        
        # Placeholder for results
        self.results_notebook = ttk.Notebook(self.results_frame)
        
        # Cost history tab
        self.cost_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.cost_tab, text="Cost History")
        
        # Coefficients tab
        self.coef_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.coef_tab, text="Coefficients")
        
        # Metrics tab
        self.metrics_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.metrics_tab, text="Metrics")

        # Regression Fit tab
        self.visualization_result_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.visualization_result_tab, text="Visualization")


    def setup_data_info_tab(self):
        frame = ttk.Frame(self.data_info_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Dataset shape
        self.shape_label = ttk.Label(frame, text="Dataset Shape: N/A", font=("Arial", 12, "bold"))
        self.shape_label.pack(anchor=tk.W, pady=(0, 10))

        # Data types section
        ttk.Label(frame, text="Column Data Types:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.dtype_frame = ttk.Frame(frame)
        self.dtype_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        # Summary statistics section
        ttk.Label(frame, text="Summary Statistics:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.summary_text = tk.Text(frame, height=12, wrap=tk.NONE)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        self.summary_text.config(state=tk.DISABLED)
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            # Try detecting separator using Python's built-in csv module
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(2048)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                sep = dialect.delimiter

            # Load CSV with detected separator
            self.data = pd.read_csv(file_path, sep=sep)
            self.processed_data = self.data.copy()
            self.columns = self.data.columns.tolist()

            # Update file label
            file_name = file_path.split("/")[-1]
            self.file_label.config(text=f"Loaded: {file_name}")

            # Refresh everything
            self.update_data_preview()
            self.update_data_info()
            self.identify_column_types()
            self.update_missing_values_tab()
            self.update_visualization_tabs()
            self.update_preprocessing_tabs()
            self.update_model_tab()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")

    def update_data_preview(self):
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Configure columns
        self.tree["columns"] = self.columns
        self.tree["show"] = "headings"
        
        for col in self.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        # Add data rows (limit to first 100 rows)
        display_data = self.processed_data.head(100)
        for i, row in display_data.iterrows():
            values = [str(row[col]) for col in self.columns]
            self.tree.insert("", "end", values=values)
    
    def update_data_info(self):
        if self.data is None:
            return

        # Update dataset shape label
        self.shape_label.config(text=f"Dataset Shape: {self.data.shape[0]} rows, {self.data.shape[1]} columns")

        # Update data types listing
        for widget in self.dtype_frame.winfo_children():
            widget.destroy()

        for i, (col, dtype) in enumerate(self.data.dtypes.items()):
            ttk.Label(self.dtype_frame, text=f"{col}: {dtype}").grid(row=i, column=0, sticky=tk.W)

        # Update summary statistics
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        summary = str(self.data.describe().round(2))
        self.summary_text.insert(tk.END, summary)
        self.summary_text.config(state=tk.DISABLED)

    
    def identify_column_types(self):
        self.categorical_columns = []
        self.numerical_columns = []
        self.boolean_columns = []
        
        for col in self.columns:
            if pd.api.types.is_bool_dtype(self.processed_data[col]):
                self.boolean_columns.append(col)
            elif pd.api.types.is_numeric_dtype(self.processed_data[col]):
                self.numerical_columns.append(col)
            else:
                # Check if it's a categorical with few unique values
                unique_values = self.processed_data[col].nunique()
                if unique_values <= 10:
                    self.categorical_columns.append(col)
                else:
                    # Text columns are treated as categorical for simplicity
                    self.categorical_columns.append(col)
    
    def update_missing_values_tab(self):
        if self.data is None:
            return
        
        # Update missing values summary
        self.missing_summary_text.config(state=tk.NORMAL)
        self.missing_summary_text.delete(1.0, tk.END)
        
        has_nulls = self.processed_data.isnull().values.any()
        null_count = self.processed_data.isnull().values.sum()
        
        summary_text = f"Dataset has missing values: {has_nulls}\n"
        summary_text += f"Total missing values: {null_count}\n"
        
        self.missing_summary_text.insert(tk.END, summary_text)
        self.missing_summary_text.config(state=tk.DISABLED)
        
        # Update missing values by column
        self.missing_columns_text.config(state=tk.NORMAL)
        self.missing_columns_text.delete(1.0, tk.END)
        
        null_by_column = self.processed_data.isnull().sum()
        columns_with_nulls = self.processed_data.columns[self.processed_data.isnull().any()]
        
        columns_text = "Missing values by column:\n"
        columns_text += str(null_by_column) + "\n\n"
        columns_text += "Columns with missing values:\n"
        columns_text += str(list(columns_with_nulls))
        
        self.missing_columns_text.insert(tk.END, columns_text)
        self.missing_columns_text.config(state=tk.DISABLED)
        
        # Update columns selection for handling missing values
        for widget in self.columns_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.missing_vars = {}
        for i, col in enumerate(self.columns):
            var = tk.BooleanVar(value=col in columns_with_nulls)
            cb = ttk.Checkbutton(self.columns_scrollable_frame, text=f"{col} ({null_by_column[col]} missing)", variable=var)
            cb.grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=5)
            self.missing_vars[col] = var
    
    def update_visualization_tabs(self):
        if self.data is None:
            return
        
        # Update target visualization dropdown
        self.target_viz_dropdown["values"] = self.numerical_columns
        if self.numerical_columns:
            self.target_viz_dropdown.current(0)
        
        # Update correlation target dropdown
        self.corr_target_dropdown["values"] = self.numerical_columns
        if self.numerical_columns:
            self.corr_target_dropdown.current(0)
        
    
    def update_preprocessing_tabs(self):
        # Update dummy variables tab
        for widget in self.dummies_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.dummy_vars = {}
        for i, col in enumerate(self.categorical_columns):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(self.dummies_scrollable_frame, text=col, variable=var)
            cb.grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=5)
            self.dummy_vars[col] = var
        
        # Update mapping tab
        for widget in self.mapping_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.mapping_vars = {}
        for i, col in enumerate(self.categorical_columns):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(self.mapping_scrollable_frame, text=col, variable=var)
            cb.grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=5)
            self.mapping_vars[col] = var
        
        # Update normalization tab
        for widget in self.normalize_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.normalize_vars = {}
        for i, col in enumerate(self.numerical_columns):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(self.normalize_scrollable_frame, text=col, variable=var)
            cb.grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=5)
            self.normalize_vars[col] = var
        
        # Update boolean tab
        for widget in self.boolean_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.boolean_vars = {}
        for i, col in enumerate(self.boolean_columns):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(self.boolean_scrollable_frame, text=col, variable=var)
            cb.grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=5)
            self.boolean_vars[col] = var
    
    def update_model_tab(self):
        # Update target variable dropdown
        self.target_dropdown["values"] = self.numerical_columns
        if self.numerical_columns:
            self.target_dropdown.current(0)
        
        # Update features checkboxes
        for widget in self.features_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.feature_vars = {}
        for i, col in enumerate(self.columns):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(self.features_scrollable_frame, text=col, variable=var)
            cb.grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=5)
            self.feature_vars[col] = var
    
    def handle_missing_values(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded.")
            return
        
        selected_columns = [col for col, var in self.missing_vars.items() if var.get()]
        
        if not selected_columns:
            messagebox.showwarning("Warning", "Please select at least one column to handle.")
            return
        
        try:
            strategy = self.strategy_var.get()
            
            if strategy == "drop":
                self.processed_data = self.processed_data.dropna(subset=selected_columns)
            else:
                for col in selected_columns:
                    if strategy == "mean" and pd.api.types.is_numeric_dtype(self.processed_data[col]):
                        self.processed_data[col] = self.processed_data[col].fillna(self.processed_data[col].mean())
                    elif strategy == "most frequent":
                        self.processed_data[col] = self.processed_data[col].fillna(self.processed_data[col].mode()[0])
                    elif strategy == "constant":
                        fill_value = self.fill_value_var.get()
                        try:
                            if pd.api.types.is_numeric_dtype(self.processed_data[col]):
                                fill_value = float(fill_value)
                        except ValueError:
                            pass
                        self.processed_data[col] = self.processed_data[col].fillna(fill_value)
            
            # Update data preview
            self.update_data_preview()
            
            # Update missing values tab
            self.update_missing_values_tab()
            
            messagebox.showinfo("Success", "Missing values handled successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to handle missing values: {str(e)}")
    
    def visualize_missing_values(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded.")
            return
        
        try:
            # Clear previous plot
            for widget in self.missing_viz_frame.winfo_children():
                widget.destroy()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot missing values heatmap
            missing_data = self.data.isnull()
            sns.heatmap(missing_data, cbar=False, cmap='viridis', ax=ax)
            ax.set_title('Missing Values Heatmap')
            ax.set_xlabel('Features')
            ax.set_ylabel('Samples')
            
            # Display plot
            canvas = FigureCanvasTkAgg(fig, master=self.missing_viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize missing values: {str(e)}")
    
    def plot_target(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded.")
            return
        
        target = self.target_viz_var.get()
        plot_type = self.target_plot_type.get()
        
        if not target:
            messagebox.showwarning("Warning", "Please select a target variable.")
            return
        
        try:
            # Clear previous plot
            for widget in self.target_plot_frame.winfo_children():
                widget.destroy()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot based on selected type
            if plot_type == "histogram":
                sns.histplot(data=self.processed_data, x=target, kde=True, ax=ax)
                ax.set_title(f'Distribution of {target}')
            # Display plot
            canvas = FigureCanvasTkAgg(fig, master=self.target_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot target: {str(e)}")
    
    def plot_correlation(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded.")
            return
        
        target = self.corr_target_var.get()
        plot_type = self.corr_plot_type.get()
        
        if not target:
            messagebox.showwarning("Warning", "Please select a target variable.")
            return
        
        try:
            # Clear previous plot
            for widget in self.corr_plot_frame.winfo_children():
                widget.destroy()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot based on selected type
            if plot_type == "heatmap":
                corr = self.processed_data.corr()
                sns.heatmap(data=corr, annot=True, cmap='coolwarm', ax=ax)
                ax.set_title('Correlation Heatmap')
            elif plot_type == "bar":
                # Drop target from dataset for correlation calculation
                dataset_without_target = self.processed_data.drop(columns=target)
                
                # Calculate correlation with target
                correlations = dataset_without_target.corrwith(self.processed_data[target])
                
                # Plot correlations
                correlations.plot.bar(ax=ax, figsize=(10, 5), grid=True)
                ax.set_title(f'Correlation with {target}')
                ax.set_ylabel('Correlation')
                ax.set_xlabel('Features')
                ax.tick_params(axis='x', rotation=90)
            
            # Display plot
            canvas = FigureCanvasTkAgg(fig, master=self.corr_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot correlation: {str(e)}")
    

    
    def apply_get_dummies(self):
        selected_columns = [col for col, var in self.dummy_vars.items() if var.get()]
        
        if not selected_columns:
            messagebox.showwarning("Warning", "Please select at least one column to convert to dummies.")
            return
        
        try:
            drop_first = self.drop_first_var.get()
            self.processed_data = pd.get_dummies(self.processed_data, columns=selected_columns, drop_first=drop_first)
            
            # Update columns and preview
            self.columns = self.processed_data.columns.tolist()
            self.update_data_preview()
            
            # Re-identify column types
            self.identify_column_types()
            
            # Update preprocessing tabs
            self.update_preprocessing_tabs()
            
            # Update model tab
            self.update_model_tab()
            
            messagebox.showinfo("Success", "Dummy variables created successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dummy variables: {str(e)}")
    
    def apply_mapping(self):
        selected_columns = [col for col, var in self.mapping_vars.items() if var.get()]

        if not selected_columns:
            messagebox.showwarning("Warning", "Please select at least one column to map.")
            return

        try:
            yes_value = self.yes_value.get().strip().lower()
            no_value = self.no_value.get().strip().lower()

            for col in selected_columns:
                self.processed_data[col] = self.processed_data[col].map(
                    lambda x: 1 if yes_value and str(x).lower() == yes_value
                    else 0 if no_value and str(x).lower() == no_value
                    else x
                )

            # Update preview
            self.update_data_preview()

            # Re-identify column types
            self.identify_column_types()

            # Update preprocessing tabs
            self.update_preprocessing_tabs()

            # Update model tab
            self.update_model_tab()

            messagebox.showinfo("Success", "Mapping applied successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply mapping: {str(e)}")

    def apply_normalization(self):
        global min_val, max_val, isNormalized
        isNormalized = True
        selected_columns = [col for col, var in self.normalize_vars.items() if var.get()]
        
        if not selected_columns:
            messagebox.showwarning("Warning", "Please select at least one column to normalize.")
            return
        
        try:
            for col in selected_columns:
                min_val = self.processed_data[col].min()
                max_val = self.processed_data[col].max()
                
                if max_val > min_val:  # Avoid division by zero
                    self.processed_data[col] = (self.processed_data[col] - min_val) / (max_val - min_val)
            
            # Update preview
            self.update_data_preview()
            
            messagebox.showinfo("Success", "Normalization applied successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply normalization: {str(e)}")
    
    def apply_boolean_conversion(self):
        selected_columns = [col for col, var in self.boolean_vars.items() if var.get()]
        
        if not selected_columns:
            messagebox.showwarning("Warning", "Please select at least one boolean column to convert.")
            return
        
        try:
            for col in selected_columns:
                self.processed_data[col] = self.processed_data[col].astype(int)
            
            # Update preview
            self.update_data_preview()
            
            # Re-identify column types
            self.identify_column_types()
            
            # Update preprocessing tabs
            self.update_preprocessing_tabs()
            
            # Update model tab
            self.update_model_tab()
            
            messagebox.showinfo("Success", "Boolean conversion applied successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to convert boolean columns: {str(e)}")
    
    def run_gradient_descent(self):
        target_col = self.target_var.get()
        selected_features = [col for col, var in self.feature_vars.items() if var.get() and col != target_col]
        
        if not target_col:
            messagebox.showwarning("Warning", "Please select a target variable.")
            return
        
        if not selected_features:
            messagebox.showwarning("Warning", "Please select at least one feature.")
            return
        
        try:
            # Prepare data
            X = self.processed_data[selected_features].copy()
            y = self.processed_data[target_col].copy()
            
            # Handle missing values
            X = X.dropna()
            y = y[X.index]
            
            # Split data
            test_size = self.test_size_var.get()
            if test_size == 0.0:
                X_train, y_train = X, y
                X_test, y_test = X.iloc[0:0], y.iloc[0:0]  # Empty test set
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Convert to numpy arrays
            X_train_np = X_train.values
            y_train_np = y_train.values
            X_test_np = X_test.values
            y_test_np = y_test.values
            
            # Add intercept if needed
            if self.add_intercept_var.get():
                X_train_np = np.column_stack((np.ones(X_train_np.shape[0]), X_train_np))
                X_test_np = np.column_stack((np.ones(X_test_np.shape[0]), X_test_np))
                selected_features = ["intercept"] + selected_features
            
            # Initialize theta
            theta = np.zeros(X_train_np.shape[1])
            
            # Run gradient descent
            learning_rate = self.learning_rate_var.get()
            iterations = self.iterations_var.get()
            theta, cost_history = self.descente_gradient(X_train_np, y_train_np, theta, learning_rate, iterations)
            
            # Make predictions
            y_train_pred = self.modele(X_train_np, theta)
            y_test_pred = self.modele(X_test_np, theta)
            
            # Calculate metrics
            train_mse = np.mean((y_train_pred - y_train_np) ** 2)
            train_r2 = self.coefficient_determination(y_train_np, y_train_pred)

            if len(y_test_np) > 0:
                test_mse = np.mean((y_test_pred - y_test_np) ** 2)
                test_r2 = self.coefficient_determination(y_test_np, y_test_pred)
            else:
                test_mse = None
                test_r2 = None
                
            # Display results
            self.display_model_results(theta, cost_history, train_mse, test_mse, train_r2, test_r2, selected_features)
            self.y_train_pred = y_train_pred
            self.y_test_pred = y_test_pred if len(y_test_np) > 0 else None
            self.y_train_actual = y_train_np
            self.y_test_actual = y_test_np if len(y_test_np) > 0 else None
            self.trained_theta = theta
            self.trained_features = selected_features
            self.create_prediction_inputs()
            self.draw_prediction_visualization()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to run gradient descent: {str(e)}")
    
    def modele(self, X, theta):
        if self.prediction_type_var.get() == "Regression":
            return X.dot(theta)
        return 1/(1+np.e**(-(X.dot(theta))))

    
    def fonction_cout(self, X, y, theta):
        m = len(y)
        if self.prediction_type_var.get() == "Regression":
            return np.sum((self.modele(X, theta) - y) ** 2) / (2 * m)
        return -1/m*np.sum(y*(np.log(self.modele(X,theta)))+(1-y)*(np.log(1-self.modele(X,theta))))
    
    def gradient(self, X, y, theta):
        m = len(y)
        if self.prediction_type_var.get() == "Regression":
            return 1/m * X.T.dot(self.modele(X, theta) - y)
        return 1/m * X.T.dot(self.modele(X, theta) - y)
    
    def descente_gradient(self, X, y, theta, alpha, nbr_iteration):
        histoire_cout = np.zeros(nbr_iteration)
        for i in range(nbr_iteration):
            theta = theta - alpha * self.gradient(X, y, theta)
            histoire_cout[i] = self.fonction_cout(X, y, theta)
        return theta, histoire_cout
    
    def coefficient_determination(self, y, y_pred):
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u/v
    
    def display_model_results(self, theta, cost_history, train_mse, test_mse, train_r2, test_r2, features):
        # Clear previous results
        for widget in self.results_notebook.winfo_children():
            for child in widget.winfo_children():
                child.destroy()
        # Clear all tabs inside the notebook instead of destroying the entire frame
        for tab in (self.cost_tab, self.coef_tab, self.metrics_tab):
            for widget in tab.winfo_children():
                widget.destroy()

        # Pack the notebook only if not already packed
        if not self.results_notebook.winfo_ismapped():
            self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Display cost history
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(range(len(cost_history)), cost_history)
        ax.set_title("Cost Function Over Iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        
        canvas = FigureCanvasTkAgg(fig, master=self.cost_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Display coefficients
        coef_frame = ttk.Frame(self.coef_tab)
        coef_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(coef_frame, text="Feature").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(coef_frame, text="Coefficient").grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        for i, (feature, coef) in enumerate(zip(features, theta)):
            ttk.Label(coef_frame, text=feature).grid(row=i+1, column=0, padx=5, pady=2, sticky=tk.W)
            ttk.Label(coef_frame, text=f"{coef:.4f}").grid(row=i+1, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Display metrics
        metrics_frame = ttk.Frame(self.metrics_tab)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(metrics_frame, text="Metric").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(metrics_frame, text="Training").grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(metrics_frame, text="Testing").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(metrics_frame, text="MSE").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(metrics_frame, text=f"{train_mse:.4f}").grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(metrics_frame, text=f"{test_mse:.4f}" if test_mse is not None else "N/A").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(metrics_frame, text="R²").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(metrics_frame, text=f"{train_r2:.4f}").grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(metrics_frame, text=f"{test_r2:.4f}" if test_r2 is not None else "N/A").grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)

    def create_prediction_inputs(self):
        for widget in self.predict_inner_frame.winfo_children():
            widget.destroy()

        self.input_vars = {}

        input_frame = ttk.Frame(self.predict_inner_frame)
        input_frame.pack(padx=10, pady=10, anchor=tk.W)

        row_index = 0
        for feature in self.trained_features:
            if feature == "intercept":
                continue
            ttk.Label(input_frame, text=feature).grid(row=row_index, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.DoubleVar()
            ttk.Entry(input_frame, textvariable=var).grid(row=row_index, column=1, padx=5, pady=5)
            self.input_vars[feature] = var
            row_index += 1

        ttk.Button(input_frame, text="Predict", command=self.predict_single_input).grid(
            row=row_index, column=0, columnspan=2, pady=10
        )

        self.prediction_result_label = ttk.Label(input_frame, text="Prediction: N/A", font=("Arial", 12, "bold"))
        self.prediction_result_label.grid(row=row_index + 1, column=0, columnspan=2, pady=10)

    def predict_single_input(self):
        try:
            values = []
            for feature in self.trained_features:
                if feature == "intercept":
                    values.append(1.0)
                else:
                    values.append(self.input_vars[feature].get())

            x_input = np.array(values)
            prediction = self.modele(x_input, self.trained_theta)
            # self.prediction_result_label.config(text=f"Prediction: {prediction:.4f}")
            if self.prediction_type_var.get() == "Regression":
                if isNormalized == True:
                    prediction = (prediction * (max_val - min_val)) + min_val
                self.prediction_result_label.config(text=f"Prediction: {prediction:.4f}")
            else:
                prediction = 1 if prediction >= 0.5 else 0
                self.prediction_result_label.config(text=f"Prediction: {'Yes' if prediction == 1 else 'No'}")    
            # Update prediction result label
        
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def draw_prediction_visualization(self):
        for widget in self.visualization_result_tab.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 4))

        if self.prediction_type_var.get() == "Regression":
            ax.scatter(self.y_train_pred, self.y_train_actual, label='Train', alpha=0.7)
            if self.y_test_pred is not None:
                ax.scatter(self.y_test_pred, self.y_test_actual, label='Test', alpha=0.7)

            min_val = min(self.y_train_actual.min(), self.y_test_actual.min() if self.y_test_actual is not None else self.y_train_actual.min())
            max_val = max(self.y_train_actual.max(), self.y_test_actual.max() if self.y_test_actual is not None else self.y_train_actual.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Regression: Predicted vs Actual")
            ax.legend()
        else:
            X = self.processed_data[self.trained_features[1:]].values
            if X.shape[1] != 2:
                ax.text(0.5, 0.5, "Need exactly 2 features for decision boundary",
                        ha='center', va='center', transform=ax.transAxes)
            else:
                x1, x2 = X[:, 0], X[:, 1]
                ax.plot(x1, x2, 'x')
                theta0, theta1, theta2 = self.trained_theta[0], self.trained_theta[1], self.trained_theta[2]
                boundary = -(theta0 + theta1 * x1) / theta2
                ax.plot(x1, boundary, color='m', label='Decision Boundary')
                ax.set_title("Classification: Decision Boundary")
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.visualization_result_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()
