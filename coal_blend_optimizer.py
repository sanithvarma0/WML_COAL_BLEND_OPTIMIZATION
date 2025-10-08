import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# FIXED: Use relative paths instead of hardcoded Windows paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'DATA')
MODELS_DIR = os.path.join(BASE_DIR, 'TRAINED_MODELS')

class CoalBlendOptimizer:
    def __init__(self, historical_data_path=None):
        """
        Initialize the coal blend optimizer with your ML inference methods
        
        Args:
            historical_data_path: Path to CSV file containing filtered historical data (40 rows)
        """
        # Target ranges for blended coal properties
        self.target_ranges = {
            'BLEND_V.M.%': (24.0, 25.0),
            'BLEND_CSN': (4.5, 6.0),
            'BLEND_Ash%': (0.0, 10.0),
            'BLEND_F.C.%': (60.0, 68.0),
            'BLEND_GM%': (9.0, 11.0),
            'BLEND_I.M.%': (0.0, 1.0)
        }
        
        # Weights for different properties (adjust based on importance)
        self.property_weights = {
            'BLEND_V.M.%': 1.0,
            'BLEND_CSN': 1.5,      # Higher weight for CSN
            'BLEND_Ash%': 2.0,     # Higher weight for Ash (critical)
            'BLEND_F.C.%': 1.0,
            'BLEND_GM%': 1.5,      # Higher weight for GM
            'BLEND_I.M.%': 1.0
        }
        
        # Load historical data
        self.historical_data = None
        self.historical_scaler = None
        if historical_data_path:
            self.load_historical_data(historical_data_path)

    # ==================== HISTORICAL DATA METHODS ====================
    def load_historical_data(self, file_path):
        """
        Load and prepare historical data for similarity matching
        
        Args:
            file_path: Path to CSV containing filtered historical data
        """
        try:
            print(f"Loading historical data from: {file_path}")
            self.historical_data = pd.read_csv(file_path)
            
            print(f"Loaded {len(self.historical_data)} historical records")
            print("Historical data columns:", list(self.historical_data.columns))
            
            # Prepare scaler for similarity matching
            self._prepare_similarity_scaler()
            
        except Exception as e:
            print(f"Warning: Could not load historical data - {e}")
            self.historical_data = None

    def rank_by_historical_priority(self, solutions):
        """
        Rank solutions with priority for historical matches over random starts
        
        Args:
            solutions: List of solution dictionaries
            
        Returns:
            Best solution with historical priority
        """
        if not solutions:
            return None
        
        # Separate historical and random solutions
        historical_solutions = [s for s in solutions if s['start_type'] == 'historical']
        random_solutions = [s for s in solutions if s['start_type'] == 'random']
        
        # Case 1: Only one historical solution - return it directly
        if len(historical_solutions) == 1:
            best_solution = historical_solutions[0]
            best_solution['ranking_method'] = 'single_historical_priority'
            best_solution['historical_priority_used'] = True
            return best_solution
        
        # Case 2: Multiple historical solutions - rank only among historical ones
        elif len(historical_solutions) > 1:
            best_solution = self.rank_by_center_values(historical_solutions)
            best_solution['ranking_method'] = 'multiple_historical_center_values'
            best_solution['historical_priority_used'] = True
            best_solution['excluded_random_solutions'] = len(random_solutions)
            return best_solution
        
        # Case 3: No historical solutions - fall back to center values ranking
        else:
            best_solution = self.rank_by_center_values(solutions)
            best_solution['ranking_method'] = 'center_values_no_historical'
            best_solution['historical_priority_used'] = False
            return best_solution

    def _prepare_similarity_scaler(self):
        """Prepare scaler for silo property similarity matching"""
        if self.historical_data is None:
            return
        
        # Define silo properties for similarity matching
        self.silo_property_columns = []
        for silo_num in [1, 2, 3, 4, 5]:
            for prop in ['Ash%', 'CSN', 'F.C.%', 'GM%', 'I.M.%', 'V.M.%']:
                col_name = f'SILO_{silo_num}_{prop}'
                if col_name in self.historical_data.columns:
                    self.silo_property_columns.append(col_name)
        
        print(f"Using {len(self.silo_property_columns)} properties for similarity matching")
        
        # Fit scaler on historical silo properties
        if self.silo_property_columns:
            historical_properties = self.historical_data[self.silo_property_columns].fillna(0)
            self.historical_scaler = StandardScaler()
            self.historical_scaler.fit(historical_properties)

    def find_historical_matches(self, silo_properties, active_silos=None, n_matches=3, verbose=True):
        """
        Find historical records with similar silo properties
        
        Args:
            silo_properties: List of 5 dictionaries containing current silo properties
            active_silos: List of active silo numbers
            n_matches: Number of closest matches to return
            verbose: Print matching details
            
        Returns:
            List of dictionaries containing matched records and their discharges
        """
        if self.historical_data is None or self.historical_scaler is None:
            if verbose:
                print("No historical data available for matching")
            return []
        
        if active_silos is None:
            active_silos = [1, 2, 3, 4, 5]
        
        try:
            # Convert current silo properties to DataFrame format
            current_row = {}
            for i, silo_dict in enumerate(silo_properties, start=1):
                current_row.update(silo_dict)
            
            current_df = pd.DataFrame([current_row])
            
            # Extract properties for similarity calculation
            current_properties = []
            for col in self.silo_property_columns:
                if col in current_df.columns:
                    current_properties.append(current_df[col].iloc[0])
                else:
                    current_properties.append(0.0)  # Default for missing properties
            
            current_properties = np.array(current_properties).reshape(1, -1)
            
            # Scale current properties
            current_scaled = self.historical_scaler.transform(current_properties)
            
            # Scale historical properties
            historical_properties = self.historical_data[self.silo_property_columns].fillna(0)
            historical_scaled = self.historical_scaler.transform(historical_properties)
            
            # Calculate distances
            distances = euclidean_distances(current_scaled, historical_scaled)[0]
            
            # Find closest matches
            closest_indices = np.argsort(distances)[:n_matches]
            
            matches = []
            for i, idx in enumerate(closest_indices):
                match_record = self.historical_data.iloc[idx]
                distance = distances[idx]
                
                # Extract discharge values from historical record
                historical_discharges = []
                for silo_num in [1, 2, 3, 4, 5]:
                    discharge_col = f'SILO_{silo_num}_DISCHARGE'
                    if discharge_col in self.historical_data.columns:
                        discharge = match_record[discharge_col]
                        # Handle inactive silos
                        if silo_num not in active_silos:
                            discharge = 0.0
                        historical_discharges.append(discharge)
                    else:
                        historical_discharges.append(0.0)
                
                # Renormalize discharges to sum to 100 for active silos
                active_discharge_sum = sum(historical_discharges[i] for i in range(5) if (i+1) in active_silos)
                if active_discharge_sum > 0:
                    for i in range(5):
                        if (i+1) in active_silos:
                            historical_discharges[i] = (historical_discharges[i] / active_discharge_sum) * 100
                        else:
                            historical_discharges[i] = 0.0
                
                matches.append({
                    'rank': i + 1,
                    'distance': distance,
                    'discharges': historical_discharges,
                    'record_index': idx,
                    'historical_record': match_record.to_dict()
                })
                
                if verbose:
                    print(f"Historical match {i+1}: Distance={distance:.4f}, "
                          f"Discharges={[f'{d:.1f}' for d in historical_discharges]}")
            
            return matches
            
        except Exception as e:
            if verbose:
                print(f"Error in historical matching: {e}")
            return []

    # ==================== ML INFERENCE METHODS ====================
    def run_inference(self, model_path, scaler_path, df, features_to_use, param):
        """Load model and scaler, then predict"""
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Ensure required features exist
            missing = [f for f in features_to_use if f not in df.columns]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            
            # Extract features in the same sequence
            X = df.loc[:, features_to_use].copy()
            
            # Ensure numeric dtype
            X = X.apply(pd.to_numeric, errors="coerce")
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Predictions
            prediction = model.predict(X_scaled)
            
            # If single-row df, return just the scalar value
            if len(prediction) == 1:
                return prediction[0]
            return prediction
            
        except Exception as e:
            print(f"Error in {param} prediction: {e}")
            return None

    def build_silo_dataframe(self, silo_properties: list, discharges: list) -> pd.DataFrame:
        """Flatten silo property dicts + discharges into a single-row DataFrame."""
        row = {}
        for i, silo_dict in enumerate(silo_properties, start=1):
            row.update(silo_dict)  # already has names like SILO_1_GM%, SILO_2_Ash%, etc.
            row[f"SILO_{i}_DISCHARGE"] = float(discharges[i-1])  # add discharge
        return pd.DataFrame([row])  # single-row DF

    def calculate_weighted_averages(self, df):
        """Calculate theoretical weighted averages for coal quality parameters"""
        result_df = df.copy()
        result_df['TOTAL_DISCHARGE'] = result_df[['SILO_1_DISCHARGE','SILO_2_DISCHARGE','SILO_3_DISCHARGE','SILO_4_DISCHARGE','SILO_5_DISCHARGE']].sum(axis=1)
        
        # Define the parameters to calculate weighted averages for
        parameters = ['Ash%', 'CSN', 'F.C.%', 'GM%', 'I.M.%', 'V.M.%']
        
        # Initialize new columns for theoretical values
        for param in parameters:
            result_df[f'THEORETICAL_{param}'] = 0.0
        
        # Process each row
        for idx, row in result_df.iterrows():
            # Get discharge values for all silos
            discharges = {
                'SILO_1': row['SILO_1_DISCHARGE'],
                'SILO_2': row['SILO_2_DISCHARGE'],
                'SILO_3': row['SILO_3_DISCHARGE'],
                'SILO_4': row['SILO_4_DISCHARGE'],
                'SILO_5': row['SILO_5_DISCHARGE']
            }
            total_discharge = row['TOTAL_DISCHARGE']
            
            # Calculate weighted average for each parameter
            for param in parameters:
                sumproduct = 0
                # Calculate SUMPRODUCT for each silo
                for silo_num in [1, 2, 3, 4, 5]:
                    silo_name = f'SILO_{silo_num}'
                    param_col = f'{silo_name}_{param}'
                    discharge = discharges[silo_name]
                    if param_col in result_df.columns and not pd.isna(row[param_col]):
                        param_value = row[param_col]
                        product = param_value * discharge
                        sumproduct += product
                
                # Calculate weighted average
                weighted_avg = sumproduct / total_discharge if total_discharge > 0 else 0
                result_df.loc[idx, f'THEORETICAL_{param}'] = weighted_avg
                
        return result_df

    def add_dominant_silo_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add dominant silo GM% (captures 'winner takes all' effect)."""
        df = df.copy()
        gm_cols = [f"SILO_{i}_GM%" for i in range(1, 6)]
        df["DOMINANT_SILO_GM"] = df[gm_cols].max(axis=1)
        return df

    def add_mixing_penalty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add GM range, discharge skew, and mixing penalty features."""
        df = df.copy()
        gm_cols = [f"SILO_{i}_GM%" for i in range(1, 6)]
        discharge_cols = [f"SILO_{i}_DISCHARGE" for i in range(1, 6)]

        df["GM_RANGE"] = df[gm_cols].max(axis=1) - df[gm_cols].min(axis=1)
        df["DISCHARGE_SKEW"] = df[discharge_cols].std(axis=1)
        df["MIXING_PENALTY"] = df["GM_RANGE"] * df["DISCHARGE_SKEW"]
        return df

    def add_deviation_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add directional deviation index (weighted by discharge)."""
        df = df.copy()
        terms = []
        for i in range(1, 6):
            terms.append(df[f"SILO_{i}_DISCHARGE"] * (df[f"SILO_{i}_GM%"] - df["THEORETICAL_GM%"]))
        df["DEVIATION_INDEX"] = sum(terms) / 100
        return df

    def gm_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature engineering steps for GM model."""
        copy_df = df.copy()
        copy_df = self.calculate_weighted_averages(copy_df)
        copy_df = self.add_dominant_silo_feature(copy_df)
        copy_df = self.add_mixing_penalty_features(copy_df)
        copy_df = self.add_deviation_index(copy_df)
        return copy_df

    def ash_engineer_features(self, df, silos=['SILO_1', 'SILO_2', 'SILO_3', 'SILO_4', 'SILO_5']):
        """Feature engineering for Ash model"""
        df_copy = df.copy()
        for s in silos:
            # Add indicator for silo usage
            df_copy[f'{s}_USED'] = (df_copy[f'{s}_DISCHARGE'] > 0).astype(int)
        df_copy = self.calculate_weighted_averages(df_copy)
        return df_copy

    def vm_engineer_features(self, df, silos=['SILO_1', 'SILO_2', 'SILO_3', 'SILO_4', 'SILO_5']):
        """Feature engineering for VM model"""
        df_copy = df.copy()
        df_copy = self.calculate_weighted_averages(df_copy)
        for s in silos:
            # Add indicator for silo usage
            df_copy[f'{s}_USED'] = (df_copy[f'{s}_DISCHARGE'] > 0).astype(int)
        return df_copy

    def compute_fc_contributions(self, df, n_silos=5, total_col="TOTAL_DISCHARGE"):
        """Compute FC contributions for each silo"""
        for i in range(1, n_silos + 1):
            df[f"SILO_{i}_F.C._contrib"] = (
                df[f"SILO_{i}_F.C.%"] * df[f"SILO_{i}_DISCHARGE"]
            )
        return df

    def compute_fc_discharge_percentages(self, df, n_silos=5, total_col="TOTAL_DISCHARGE"):
        """Compute discharge percentages for FC model"""
        for i in range(1, n_silos + 1):
            df[f"SILO_{i}_DISCHARGE%"] = df[f"SILO_{i}_DISCHARGE"] / df[total_col]
        return df

    def process_fc_silo_data(self, df, n_silos=5, total_col="TOTAL_DISCHARGE"):
        """Process silo data for FC model"""
        df = self.compute_fc_contributions(df, n_silos, total_col)
        df = self.compute_fc_discharge_percentages(df, n_silos, total_col)
        return df

    def fc_engineer_features(self, df, silos=['SILO_1', 'SILO_2', 'SILO_3', 'SILO_4', 'SILO_5']):
        """Feature engineering for FC model"""
        df_copy = df.copy()
        df_copy['TOTAL_DISCHARGE'] = df_copy[['SILO_1_DISCHARGE','SILO_2_DISCHARGE','SILO_3_DISCHARGE','SILO_4_DISCHARGE','SILO_5_DISCHARGE']].sum(axis=1)
        df_copy = self.process_fc_silo_data(df_copy)
        return df_copy

    # ==================== INDIVIDUAL MODEL PREDICTIONS ====================
    def predict_gm_model(self, silo_properties, discharges):
        """GM% prediction - FIXED with relative paths"""
        gm_model_path = os.path.join(MODELS_DIR, 'GM', 'GM_RIDGE_MODEL_V1.joblib')
        gm_scaler_path = os.path.join(MODELS_DIR, 'GM', 'GM_RIDGE_SCALER_V1.joblib')
        gm_features_to_use = ['SILO_1_GM%', 'SILO_2_GM%', 'SILO_3_GM%', 'SILO_4_GM%', 'SILO_5_GM%', 
                             'THEORETICAL_GM%', 'DEVIATION_INDEX', 'DOMINANT_SILO_GM', 'MIXING_PENALTY', 
                             'GM_RANGE', 'DISCHARGE_SKEW']
        
        gm_df = self.build_silo_dataframe(silo_properties, discharges)
        gm_silo_properties = self.gm_engineer_features(gm_df)
        predicted_gm = self.run_inference(gm_model_path, gm_scaler_path, gm_silo_properties, gm_features_to_use, 'GM%')
        return predicted_gm

    def predict_ash_model(self, silo_properties, discharges):
        """Ash% prediction - FIXED with relative paths"""
        ash_model_path = os.path.join(MODELS_DIR, 'ASH', 'ASH_RIDGE_MODEL_V1.joblib')
        ash_scaler_path = os.path.join(MODELS_DIR, 'ASH', 'ASH_RIDGE_SCALER_V1.joblib')
        ash_features_to_use = ['SILO_1_Ash%', 'SILO_2_Ash%', 'SILO_3_Ash%', 'SILO_4_Ash%', 'SILO_5_Ash%', 
                              'SILO_1_DISCHARGE', 'SILO_2_DISCHARGE', 'SILO_3_DISCHARGE', 'SILO_4_DISCHARGE', 'SILO_5_DISCHARGE', 
                              'SILO_1_USED', 'SILO_2_USED', 'SILO_3_USED', 'SILO_4_USED', 'SILO_5_USED', 'THEORETICAL_Ash%']
        
        ash_df = self.build_silo_dataframe(silo_properties, discharges)
        ash_silo_properties = self.ash_engineer_features(ash_df)
        predicted_ash = self.run_inference(ash_model_path, ash_scaler_path, ash_silo_properties, ash_features_to_use, 'Ash%')
        return predicted_ash

    def predict_im_model(self, silo_properties, discharges):
        """IM% prediction - FIXED with relative paths"""
        im_model_path = os.path.join(MODELS_DIR, 'IM', 'IM_RIDGE_MODEL_V1.joblib')
        im_scaler_path = os.path.join(MODELS_DIR, 'IM', 'IM_RIDGE_SCALER_V1.joblib')
        im_features_to_use = ['SILO_1_I.M.%', 'SILO_2_I.M.%', 'SILO_3_I.M.%', 'SILO_4_I.M.%', 'SILO_5_I.M.%', 
                             'SILO_1_DISCHARGE', 'SILO_2_DISCHARGE', 'SILO_3_DISCHARGE', 'SILO_4_DISCHARGE', 'SILO_5_DISCHARGE']
        
        im_df = self.build_silo_dataframe(silo_properties, discharges)
        im_silo_properties = im_df.copy()
        predicted_im = self.run_inference(im_model_path, im_scaler_path, im_silo_properties, im_features_to_use, 'IM%')
        return predicted_im

    def predict_vm_model(self, silo_properties, discharges):
        """VM% prediction - FIXED with relative paths"""
        vm_model_path = os.path.join(MODELS_DIR, 'VM', 'VM_LINEAR_MODEL_V1.joblib')
        vm_scaler_path = os.path.join(MODELS_DIR, 'VM', 'VM_LINEAR_SCALER_V1.joblib')
        vm_features_to_use = ['SILO_1_V.M.%', 'SILO_2_V.M.%', 'SILO_3_V.M.%', 'SILO_4_V.M.%', 'SILO_5_V.M.%', 
                             'SILO_1_DISCHARGE', 'SILO_2_DISCHARGE', 'SILO_3_DISCHARGE', 'SILO_4_DISCHARGE', 'SILO_5_DISCHARGE', 
                             'SILO_1_USED', 'SILO_2_USED', 'SILO_3_USED', 'SILO_4_USED', 'SILO_5_USED', 'THEORETICAL_V.M.%']
        
        vm_df = self.build_silo_dataframe(silo_properties, discharges)
        vm_silo_properties = self.vm_engineer_features(vm_df)
        predicted_vm = self.run_inference(vm_model_path, vm_scaler_path, vm_silo_properties, vm_features_to_use, 'VM%')
        return predicted_vm

    def predict_fc_model(self, silo_properties, discharges):
        """FC% prediction - FIXED with relative paths"""
        fc_model_path = os.path.join(MODELS_DIR, 'FC', 'FC_LINEAR_MODEL_V1.joblib')
        fc_scaler_path = os.path.join(MODELS_DIR, 'FC', 'FC_LINEAR_SCALER_V1.joblib')
        fc_features_to_use = ["SILO_1_F.C._contrib", "SILO_2_F.C._contrib", "SILO_3_F.C._contrib", 
                             "SILO_4_F.C._contrib", "SILO_5_F.C._contrib", "SILO_1_DISCHARGE%", 
                             "SILO_2_DISCHARGE%", "SILO_3_DISCHARGE%", "SILO_4_DISCHARGE%", "SILO_5_DISCHARGE%"]
        
        fc_df = self.build_silo_dataframe(silo_properties, discharges)
        fc_silo_properties = self.fc_engineer_features(fc_df)
        predicted_fc = self.run_inference(fc_model_path, fc_scaler_path, fc_silo_properties, fc_features_to_use, 'FC%')
        return predicted_fc

    def predict_csn_model(self, silo_properties, discharges):
        """CSN prediction - FIXED with relative paths"""
        csn_model_path = os.path.join(MODELS_DIR, 'CSN', 'CSN_LINEAR_MODEL_V1.joblib')
        csn_scaler_path = os.path.join(MODELS_DIR, 'CSN', 'CSN_LINEAR_SCALER_V1.joblib')
        csn_features_to_use = ['SILO_1_CSN', 'SILO_2_CSN', 'SILO_3_CSN', 'SILO_4_CSN', 'SILO_5_CSN']
        
        csn_df = self.build_silo_dataframe(silo_properties, discharges)
        csn_silo_properties = csn_df.copy()
        predicted_csn = self.run_inference(csn_model_path, csn_scaler_path, csn_silo_properties, csn_features_to_use, 'CSN')
        return predicted_csn

    # ==================== OPTIMIZATION CORE ====================
    def predict_all_blend_properties(self, silo_properties, discharges, active_silos=None, verbose=False):
        """
        Predict all blended coal properties using your ML models
        
        Args:
            silo_properties: List of 5 dictionaries containing silo properties
            discharges: List of 5 discharge values
            active_silos: List of active silo numbers (1-5)
            verbose: Print detailed predictions
            
        Returns:
            Dictionary of predicted blend properties
        """
        if active_silos is None:
            active_silos = [1, 2, 3, 4, 5]
        
        # Set inactive silo properties and discharges to 0
        adjusted_silo_properties = []
        adjusted_discharges = list(discharges)
        
        for i in range(5):
            silo_num = i + 1
            if silo_num in active_silos:
                adjusted_silo_properties.append(silo_properties[i])
            else:
                # Create zero properties for inactive silo
                zero_properties = {k: 0.0 for k in silo_properties[i].keys()}
                adjusted_silo_properties.append(zero_properties)
                adjusted_discharges[i] = 0.0
        
        try:
            predictions = {}
            
            # Predict each property
            predictions['BLEND_GM%'] = self.predict_gm_model(adjusted_silo_properties, adjusted_discharges)
            predictions['BLEND_Ash%'] = self.predict_ash_model(adjusted_silo_properties, adjusted_discharges)
            predictions['BLEND_I.M.%'] = self.predict_im_model(adjusted_silo_properties, adjusted_discharges)
            predictions['BLEND_V.M.%'] = self.predict_vm_model(adjusted_silo_properties, adjusted_discharges)
            predictions['BLEND_F.C.%'] = self.predict_fc_model(adjusted_silo_properties, adjusted_discharges)
            predictions['BLEND_CSN'] = self.predict_csn_model(adjusted_silo_properties, adjusted_discharges)
            
            # Filter out None values (failed predictions)
            predictions = {k: v for k, v in predictions.items() if v is not None}
            
            if verbose:
                print(f"Predictions: {predictions}")
            
            return predictions
            
        except Exception as e:
            if verbose:
                print(f"Error in prediction: {e}")
            return {}

    def objective_function(self, discharges, silo_properties, active_silos=None, verbose=False):
        """
        Enhanced objective function for SLSQP optimization with progressive penalties
        """
        try:
            # Early termination for invalid discharges
            if np.any(discharges < 0) or abs(np.sum(discharges) - 100) > 1e-6:
                return 1e8
            
            # Predict blend properties
            predictions = self.predict_all_blend_properties(
                silo_properties, discharges, active_silos, verbose=verbose
            )
            
            if not predictions:  # Failed predictions
                return 1e6
            
            penalty = 0.0
            violations = {}
            
            # Calculate weighted penalties for each property with progressive penalty
            for prop_name, (min_val, max_val) in self.target_ranges.items():
                if prop_name in predictions:
                    pred_val = predictions[prop_name]
                    weight = self.property_weights.get(prop_name, 1.0)
                    
                    if pred_val < min_val:
                        violation = min_val - pred_val
                        # Progressive penalty: gentle near boundary, harsh for large violations
                        if violation > 0.5:
                            penalty_val = 0.5 + 10 * (violation - 0.5)
                        else:
                            penalty_val = violation
                        penalty += weight * (penalty_val ** 2)
                        violations[prop_name] = f"Below target: {pred_val:.2f} < {min_val}"
                        
                    elif pred_val > max_val:
                        violation = pred_val - max_val
                        # Progressive penalty: gentle near boundary, harsh for large violations
                        if violation > 0.5:
                            penalty_val = 0.5 + 10 * (violation - 0.5)
                        else:
                            penalty_val = violation
                        penalty += weight * (penalty_val ** 2)
                        violations[prop_name] = f"Above target: {pred_val:.2f} > {max_val}"
            
            if verbose and violations:
                print(f"Violations: {violations}")
                print(f"Total penalty: {penalty:.4f}")
            
            return penalty
            
        except Exception as e:
            if verbose:
                print(f"Error in objective function: {e}")
            return 1e6

    def optimize_blend(self, silo_properties, active_silos=None, initial_guess=None, 
                      discharge_bounds=(0, 100), verbose=True):
        """
        Optimize discharge values using SLSQP with enhanced constraint handling
        """
        if active_silos is None:
            active_silos = [1, 2, 3, 4, 5]
        
        n_silos = 5
        
        # Initial guess - equal distribution among active silos
        if initial_guess is None:
            initial_guess = np.zeros(n_silos)
            active_discharge = 100.0 / len(active_silos)
            for silo_idx in range(n_silos):
                if (silo_idx + 1) in active_silos:
                    initial_guess[silo_idx] = active_discharge
        
        # Set bounds
        bounds = []
        for silo_idx in range(n_silos):
            if (silo_idx + 1) in active_silos:
                bounds.append(discharge_bounds)
            else:
                bounds.append((0, 0))  # Inactive silos must be 0
        
        # Enhanced constraint: sum of discharges = 100
        def discharge_sum_constraint(x):
            return np.sum(x) - 100.0
        
        constraints = [{
            'type': 'eq',
            'fun': discharge_sum_constraint,
            'jac': lambda x: np.ones(5)  # Gradient for faster convergence
        }]
        
        if verbose:
            print(f"Starting optimization...")
            print(f"Active silos: {active_silos}")
            print(f"Initial guess: {initial_guess}")
            print(f"Bounds: {bounds}")
        
        # SLSQP Optimization
        result = minimize(
            fun=lambda x: self.objective_function(x, silo_properties, active_silos, verbose=False),
            x0=initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 1000,
                'ftol': 1e-9,
                'disp': verbose
            }
        )
        
        # Ensure exact constraint satisfaction (handle floating point errors)
        if result.success and result.x is not None:
            result.x = result.x * (100.0 / np.sum(result.x))
        
        return result

    def rank_by_center_values(self, solutions):
        """
        Rank solutions by proximity to target range centers
        Lower score is better (closer to centers)
        
        Args:
            solutions: List of solution dictionaries
            
        Returns:
            Best solution (closest to target centers)
        """
        if not solutions:
            return None
        
        best_solution = None
        best_score = float('inf')
        
        for solution in solutions:
            predictions = solution['evaluation']['predictions']
            center_deviation_score = 0.0
            
            # Calculate deviation from center for each parameter
            for prop_name, (min_val, max_val) in self.target_ranges.items():
                if prop_name in predictions:
                    center = (min_val + max_val) / 2
                    pred_val = predictions[prop_name]
                    
                    # Calculate normalized deviation from center
                    range_size = max_val - min_val
                    deviation = abs(pred_val - center) / range_size if range_size > 0 else 0
                    
                    # Weight by property importance
                    weight = self.property_weights.get(prop_name, 1.0)
                    center_deviation_score += weight * deviation
            
            # Track best solution
            if center_deviation_score < best_score:
                best_score = center_deviation_score
                best_solution = solution
                
        # Add ranking info to the best solution
        if best_solution:
            best_solution['ranking_method'] = 'center_values'
            best_solution['center_deviation_score'] = best_score
            
        return best_solution

    def select_best_solution_case4(self, solutions):
        """
        Select best solution for Case 4 based on satisfied_count and objective_value
        
        Args:
            solutions: List of solution dictionaries
            
        Returns:
            Best solution based on satisfaction count first, then objective value
        """
        if not solutions:
            return None
        
        # Sort by satisfied_count (descending), then by objective_value (ascending)
        sorted_solutions = sorted(solutions, 
                                key=lambda x: (-x['satisfied_count'], x['objective_value']))
        
        best_solution = sorted_solutions[0]
        best_solution['ranking_method'] = 'satisfaction_count_then_objective'
        
        return best_solution

    def count_satisfied_parameters(self, predictions):
        """
        Count how many parameters are within their target ranges
        
        Returns:
            Tuple: (satisfied_count, satisfied_params, violated_params)
        """
        satisfied_params = []
        violated_params = []
        
        for prop_name, (min_val, max_val) in self.target_ranges.items():
            if prop_name in predictions:
                pred_val = predictions[prop_name]
                if min_val <= pred_val <= max_val:
                    satisfied_params.append(prop_name)
                else:
                    violated_params.append(prop_name)
        
        return len(satisfied_params), satisfied_params, violated_params

    def multi_start_optimization_with_historical(self, silo_properties, active_silos=None, 
                                                n_historical_starts=3, n_random_starts=3, verbose=True):
        """
        Enhanced multi-start optimization using historical data + random initialization
        Returns ALL valid solutions categorized by satisfaction level
        
        Args:
            silo_properties: List of 5 dictionaries with silo properties
            active_silos: List of active silo numbers
            n_historical_starts: Number of historical matches to use as starting points
            n_random_starts: Number of random starting points
            verbose: Print detailed progress
            
        Returns:
            Dictionary containing all solutions categorized by satisfaction level
        """
        if active_silos is None:
            active_silos = [1, 2, 3, 4, 5]
        
        all_results = []
        all_valid_solutions = []
        
        if verbose:
            print(f"="*80)
            print(f"ENHANCED MULTI-START OPTIMIZATION - ALL SOLUTIONS")
            print(f"="*80)
            print(f"Historical starts: {n_historical_starts}, Random starts: {n_random_starts}")
            print(f"Active silos: {active_silos}")
            print(f"")
        
        # PHASE 1: Historical matches as starting points
        historical_matches = self.find_historical_matches(
            silo_properties, active_silos, n_matches=n_historical_starts, verbose=verbose
        )
        
        if verbose and historical_matches:
            print(f"\nPHASE 1: HISTORICAL STARTING POINTS")
            print(f"-" * 50)
        
        for i, match in enumerate(historical_matches):
            start_name = f"Historical-{i+1}"
            initial_guess = np.array(match['discharges'])
            
            if verbose:
                print(f"{start_name}: Distance={match['distance']:.4f}, "
                      f"Discharges={[f'{d:.1f}' for d in initial_guess]}")
            
            # Run optimization
            result = self.optimize_blend(
                silo_properties, active_silos, initial_guess, verbose=False
            )
            
            if result.success:
                # Evaluate the solution
                evaluation = self.evaluate_solution(silo_properties, result.x, active_silos)
                satisfied_count, satisfied_params, violated_params = self.count_satisfied_parameters(evaluation['predictions'])
                
                solution_info = {
                    'start_type': 'historical',
                    'start_name': start_name,
                    'initial_guess': initial_guess.copy(),
                    'optimal_discharges': result.x.copy(),
                    'objective_value': result.fun,
                    'evaluation': evaluation,
                    'satisfied_count': satisfied_count,
                    'satisfied_params': satisfied_params,
                    'violated_params': violated_params,
                    'historical_distance': match['distance'],
                    'historical_rank': match['rank'],
                    'scipy_result': result
                }
                all_valid_solutions.append(solution_info)
                
                if verbose:
                    print(f"    ✅ Score: {result.fun:.6f}, Satisfied: {satisfied_count}/6 parameters")
            elif verbose:
                print(f"    ❌ Optimization failed")
            
            all_results.append({
                'start_type': 'historical',
                'start_name': start_name,
                'result': result,
                'historical_info': match
            })
        
        # PHASE 2: Random starting points
        if verbose:
            print(f"\nPHASE 2: RANDOM STARTING POINTS")
            print(f"-" * 50)
        
        for start_idx in range(n_random_starts):
            start_name = f"Random-{start_idx + 1}"
            
            # Generate random initial guess
            initial_guess = np.random.uniform(5, 40, 5)  # Random between 5-40%
            
            # Set inactive silos to 0
            for silo_idx in range(5):
                if (silo_idx + 1) not in active_silos:
                    initial_guess[silo_idx] = 0
            
            # Normalize active silos to sum to 100
            active_sum = sum(initial_guess[i] for i in range(5) if (i+1) in active_silos)
            if active_sum > 0:
                for silo_idx in range(5):
                    if (silo_idx + 1) in active_silos:
                        initial_guess[silo_idx] = (initial_guess[silo_idx] / active_sum) * 100
                    else:
                        initial_guess[silo_idx] = 0.0
            
            if verbose:
                print(f"{start_name}: Discharges={[f'{d:.1f}' for d in initial_guess]}")
            
            # Run optimization
            result = self.optimize_blend(
                silo_properties, active_silos, initial_guess, verbose=False
            )
            
            if result.success:
                # Evaluate the solution
                evaluation = self.evaluate_solution(silo_properties, result.x, active_silos)
                satisfied_count, satisfied_params, violated_params = self.count_satisfied_parameters(evaluation['predictions'])
                
                solution_info = {
                    'start_type': 'random',
                    'start_name': start_name,
                    'initial_guess': initial_guess.copy(),
                    'optimal_discharges': result.x.copy(),
                    'objective_value': result.fun,
                    'evaluation': evaluation,
                    'satisfied_count': satisfied_count,
                    'satisfied_params': satisfied_params,
                    'violated_params': violated_params,
                    'historical_distance': None,
                    'historical_rank': None,
                    'scipy_result': result
                }
                all_valid_solutions.append(solution_info)
                
                if verbose:
                    print(f"    ✅ Score: {result.fun:.6f}, Satisfied: {satisfied_count}/6 parameters")
            elif verbose:
                print(f"    ❌ Optimization failed")
            
            all_results.append({
                'start_type': 'random',
                'start_name': start_name,
                'result': result,
                'historical_info': None
            })
        
        # CATEGORIZE SOLUTIONS
        solutions_all_satisfied = [s for s in all_valid_solutions if s['satisfied_count'] == 6]
        solutions_three_plus = [s for s in all_valid_solutions if s['satisfied_count'] >= 3]
        
        if verbose:
            print(f"\n" + "="*80)
            print(f"SOLUTION CATEGORIZATION SUMMARY")
            print(f"="*80)
            print(f"Total successful optimizations: {len(all_valid_solutions)}")
            print(f"Solutions satisfying ALL 6 parameters: {len(solutions_all_satisfied)}")
            print(f"Solutions satisfying 3+ parameters: {len(solutions_three_plus)}")
        
        # Prepare return structure
        result_summary = {
            'all_solutions': all_valid_solutions,
            'solutions_all_satisfied': solutions_all_satisfied,
            'solutions_three_plus': solutions_three_plus,
            'all_optimization_attempts': all_results,
            'total_attempts': len(all_results),
            'successful_attempts': len(all_valid_solutions)
        }
        
        return result_summary

    def multi_start_optimization(self, silo_properties, active_silos=None, n_starts=6, verbose=True):
        """
        Legacy method - now redirects to enhanced version for backward compatibility
        """
        n_hist = min(3, n_starts//2)
        n_rand = n_starts - n_hist
        return self.multi_start_optimization_with_historical(
            silo_properties, active_silos, 
            n_historical_starts=n_hist, 
            n_random_starts=n_rand, 
            verbose=verbose
        )

    def evaluate_solution(self, silo_properties, discharges, active_silos=None):
        """
        Evaluate a discharge solution and check feasibility
        
        Returns:
            Dictionary with predictions, feasibility status, and violations
        """
        predictions = self.predict_all_blend_properties(
            silo_properties, discharges, active_silos, verbose=True
        )
        
        feasible = True
        violations = {}
        
        # Check each constraint
        for prop_name, (min_val, max_val) in self.target_ranges.items():
            if prop_name in predictions:
                pred_val = predictions[prop_name]
                if pred_val < min_val or pred_val > max_val:
                    feasible = False
                    violations[prop_name] = {
                        'predicted': pred_val,
                        'target_range': (min_val, max_val),
                        'violation_amount': min(abs(pred_val - min_val), abs(pred_val - max_val))
                    }
        
        return {
            'predictions': predictions,
            'feasible': feasible,
            'violations': violations,
            'total_discharge': np.sum(discharges),
            'active_silos': active_silos or [1,2,3,4,5]
        }

# ==================== MAIN OPTIMIZATION FUNCTION ====================
def optimize_coal_blend_enhanced(silo_properties, n_historical_starts, n_random_starts, active_silos=None, 
                                use_enhanced_multi_start=True, verbose=True):
    """
    Enhanced main function to optimize coal blend discharges with all-solutions approach
    FIXED FOR STREAMLIT CLOUD DEPLOYMENT
    
    Returns solutions categorized by satisfaction level following the 4-case logic:
    Case 1: Single combination satisfying all 6 parameters
    Case 2: Multiple combinations satisfying all 6 parameters  
    Case 3: Single combination satisfying 3+ parameters (when no all-6 solution exists)
    Case 4: Multiple combinations satisfying 3+ parameters (when no all-6 solution exists)
    
    Args:
        silo_properties: List of 5 dictionaries containing silo properties
        n_historical_starts: Number of historical starting points
        n_random_starts: Number of random starting points
        active_silos: List of active silo numbers [1,2,3,4,5] or subset
        use_enhanced_multi_start: Use historical + random starting points
        verbose: Print detailed progress
        
    Returns:
        Dictionary with solutions categorized by case logic
    """
    # FIXED: Use relative path instead of hardcoded Windows path
    historical_data_path = os.path.join(DATA_DIR, 'HISTORICAL_DATA_COAL_BLEND_OPTIMIZATION_V2.csv')
    
    # Check if historical data file exists
    if not os.path.exists(historical_data_path):
        print(f"Warning: Historical data file not found at {historical_data_path}")
        print(f"Expected location: {os.path.abspath(historical_data_path)}")
        historical_data_path = None
    
    # Initialize optimizer with historical data
    optimizer = CoalBlendOptimizer(historical_data_path=historical_data_path)
    
    if verbose:
        print("="*100)
        print("ENHANCED COAL BLEND OPTIMIZATION - ALL SOLUTIONS APPROACH")
        print("="*100)
        print(f"Active silos: {active_silos or [1,2,3,4,5]}")
        print(f"Target ranges: {optimizer.target_ranges}")
        print(f"Historical data loaded: {'Yes' if optimizer.historical_data is not None else 'No'}")
        if optimizer.historical_data is not None:
            print(f"Historical records available: {len(optimizer.historical_data)}")
    
    # Run enhanced optimization to get ALL solutions
    if use_enhanced_multi_start:
        optimization_summary = optimizer.multi_start_optimization_with_historical(
            silo_properties, active_silos, n_historical_starts, n_random_starts, verbose=verbose
        )
    else:
        # For single optimization, wrap in similar structure
        result = optimizer.optimize_blend(silo_properties, active_silos, verbose=verbose)
        if result.success:
            evaluation = optimizer.evaluate_solution(silo_properties, result.x, active_silos)
            satisfied_count, satisfied_params, violated_params = optimizer.count_satisfied_parameters(evaluation['predictions'])
            
            single_solution = {
                'start_type': 'single',
                'start_name': 'Single-Optimization',
                'optimal_discharges': result.x.copy(),
                'objective_value': result.fun,
                'evaluation': evaluation,
                'satisfied_count': satisfied_count,
                'satisfied_params': satisfied_params,
                'violated_params': violated_params,
                'scipy_result': result
            }
            
            optimization_summary = {
                'all_solutions': [single_solution],
                'solutions_all_satisfied': [single_solution] if satisfied_count == 6 else [],
                'solutions_three_plus': [single_solution] if satisfied_count >= 3 else [],
                'total_attempts': 1,
                'successful_attempts': 1
            }
        else:
            optimization_summary = {
                'all_solutions': [],
                'solutions_all_satisfied': [],
                'solutions_three_plus': [],
                'total_attempts': 1,
                'successful_attempts': 0
            }
    
    # Apply Case Logic with ranking
    solutions_all_satisfied = optimization_summary['solutions_all_satisfied']
    solutions_three_plus = optimization_summary['solutions_three_plus']
    
    case_result = None
    case_type = None
    
    if len(solutions_all_satisfied) == 1:
        # CASE 1: Single combination satisfying all parameters
        case_type = "Case 1: Single solution satisfying ALL 6 parameters"
        case_result = solutions_all_satisfied[0]
        
    elif len(solutions_all_satisfied) > 1:
        # CASE 2: Multiple combinations satisfying all parameters - Apply historical priority ranking
        case_type = "Case 2: Best solution from multiple satisfying ALL 6 parameters (historical priority)"
        case_result = optimizer.rank_by_historical_priority(solutions_all_satisfied)
        # Store all solutions for reference
        case_result['all_perfect_solutions'] = solutions_all_satisfied
        case_result['total_perfect_solutions'] = len(solutions_all_satisfied)
        
    elif len(solutions_three_plus) == 1:
        # CASE 3: Single combination satisfying 3+ parameters
        case_type = "Case 3: Single solution satisfying 3+ parameters (no all-6 solution found)"
        case_result = solutions_three_plus[0]
        
    elif len(solutions_three_plus) > 1:
        # CASE 4: Multiple combinations satisfying 3+ parameters - Select best
        case_type = "Case 4: Best solution from multiple satisfying 3+ parameters (ranked by satisfaction count)"
        case_result = optimizer.select_best_solution_case4(solutions_three_plus)
        # Store all solutions for reference
        case_result['all_partial_solutions'] = solutions_three_plus
        case_result['total_partial_solutions'] = len(solutions_three_plus)
        
    else:
        # No acceptable solutions found
        case_type = "No solutions found satisfying 3+ parameters"
        case_result = None
    
    if verbose:
        print(f"\n" + "="*100)
        print(f"CASE DETERMINATION AND RESULTS")
        print(f"="*100)
        print(f"Determined: {case_type}")
        print(f"Total successful optimizations: {optimization_summary['successful_attempts']}")
        print(f"Solutions with all 6 satisfied: {len(solutions_all_satisfied)}")
        print(f"Solutions with 3+ satisfied: {len(solutions_three_plus)}")
        print()
        
        # Print detailed results based on case
        if case_result is None:
            print("NO ACCEPTABLE SOLUTIONS FOUND")
            print("   No solutions satisfied at least 3 parameters")
            
        else:
            # Single solution for all cases now
            solution = case_result
            print(f"SELECTED SOLUTION: {solution['start_name']} ({solution['start_type']})")
            print("=" * 80)
            print(f"Objective Score: {solution['objective_value']:.6f}")
            print(f"Parameters Satisfied: {solution['satisfied_count']}/6")
            print(f"Satisfied: {', '.join(solution['satisfied_params'])}")
            if solution['violated_params']:
                print(f"Violated: {', '.join(solution['violated_params'])}")
            
            # Show ranking information
            if solution.get('ranking_method'):
                if solution['ranking_method'] == 'center_values':
                    print(f"Ranking Method: Center Values (deviation score: {solution['center_deviation_score']:.6f})")
                    if solution.get('total_perfect_solutions'):
                        print(f"Selected from {solution['total_perfect_solutions']} perfect solutions")
                elif solution['ranking_method'] == 'satisfaction_count_then_objective':
                    print(f"Ranking Method: Satisfaction Count then Objective Value")
                    if solution.get('total_partial_solutions'):
                        print(f"Selected from {solution['total_partial_solutions']} partial solutions")
            
            print(f"\nOptimal Discharge Values:")
            for i, discharge in enumerate(solution['optimal_discharges'], 1):
                status = "Active" if i in (active_silos or [1,2,3,4,5]) else "Inactive"
                print(f"  SILO_{i}_DISCHARGE: {discharge:6.2f}% ({status})")
            
            print(f"\nPredicted Blend Properties:")
            for prop, value in solution['evaluation']['predictions'].items():
                target_min, target_max = optimizer.target_ranges[prop]
                center = (target_min + target_max) / 2
                
                if target_min <= value <= target_max:
                    status = "OK"
                    deviation_from_center = abs(value - center)
                    indicator = f" (center deviation: {deviation_from_center:.2f})"
                else:
                    status = "VIOLATION"
                    if value < target_min:
                        indicator = f" (below by {target_min - value:.2f})"
                    else:
                        indicator = f" (above by {value - target_max:.2f})"
                
                print(f"  {prop:15s}: {value:6.2f} (target: {target_min:5.1f}-{target_max:5.1f}, center: {center:5.1f}) {status}{indicator}")
            
            if solution.get('historical_distance') is not None:
                print(f"\nHistorical Match Distance: {solution['historical_distance']:.4f}")

            
            # Show historical priority information
            if solution.get('historical_priority_used'):
                if solution['ranking_method'] == 'single_historical_priority':
                    print(f"Selection: Single historical solution prioritized over {len(solution.get('all_perfect_solutions', [])) - 1} random solutions")
                elif solution['ranking_method'] == 'multiple_historical_center_values':
                    print(f"Selection: Best historical solution (excluded {solution.get('excluded_random_solutions', 0)} random solutions)")
                
                if solution.get('historical_distance') is not None:
                    print(f"Historical Match Distance: {solution['historical_distance']:.4f}")
            
            # Show alternative solutions summary for Case 2 and Case 4
            if solution.get('all_perfect_solutions'):
                print(f"\nALTERNATIVE PERFECT SOLUTIONS SUMMARY:")
                print(f"Total perfect solutions found: {len(solution['all_perfect_solutions'])}")
                for i, alt_sol in enumerate(solution['all_perfect_solutions'], 1):
                    if alt_sol['start_name'] != solution['start_name']:  # Don't repeat selected solution
                        print(f"  Alternative {i}: {alt_sol['start_name']} (score: {alt_sol['objective_value']:.6f})")
                        
            elif solution.get('all_partial_solutions'):
                print(f"\nALTERNATIVE PARTIAL SOLUTIONS SUMMARY:")
                print(f"Total partial solutions found: {len(solution['all_partial_solutions'])}")
                for i, alt_sol in enumerate(solution['all_partial_solutions'], 1):
                    if alt_sol['start_name'] != solution['start_name']:  # Don't repeat selected solution
                        print(f"  Alternative {i}: {alt_sol['start_name']} ({alt_sol['satisfied_count']}/6 satisfied, score: {alt_sol['objective_value']:.6f})")
    
    # Return comprehensive result with corrected structure
    return {
        'success': case_result is not None,
        'case_type': case_type,
        'selected_solution': case_result,
        'optimization_summary': optimization_summary,
        'historical_data_used': optimizer.historical_data is not None,
        'active_silos': active_silos or [1,2,3,4,5],
        'total_attempts': optimization_summary['total_attempts'],
        'successful_attempts': optimization_summary['successful_attempts']
    }


