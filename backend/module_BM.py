import pandas as pd
from itertools import combinations
from core_config import PARAMS_MAIN_MAX_ITERATION, VERBOSE
from module_transform import Transform
from eval import Evaluator

class BiasMitigation:
    def __init__(self, evaluator: Evaluator, transformer: Transform, label_O, cate_attrs, num_attrs):
        """
        Initialize BiasMitigation class
        
        Parameters:
        evaluator: Evaluator
            Evaluator instance for calculating metrics
        transformer: Transform
            Transform instance for data transformation
        label_O: str or list of str
            Protected attribute(s) to mitigate bias
        label_Y: str
            Target variable
        cate_attrs: list of str
            Categorical attributes in the dataset
        num_attrs: list of str
            Numerical attributes in the dataset
        """
        self.label_O = label_O
        self.cate_attrs = cate_attrs
        self.num_attrs = num_attrs
        self.evaluator = evaluator
        self.transformer = transformer
    
    def _find_max_epsilon_attribute(self, df_epsilon):
        """
        Find the attribute with maximum epsilon value from df_epsilon
        
        Parameters:
        df_epsilon: dict
            Epsilon values from Evaluator.calculate_epsilon
            
        Returns:
        tuple
            (selected_label_O, selected_attribute) - The attribute pair with highest epsilon
        """
        # Find the maximum epsilon value across all label_O and attributes
        max_epsilon = -1
        selected_label_O = None
        selected_attribute = None
        
        # Iterate through each protected attribute
        for label_O in df_epsilon:
            # Get the epsilon values for this protected attribute
            epsilon_series = df_epsilon[label_O]
            
            # Find the attribute with maximum epsilon for this label_O
            max_attr = epsilon_series.idxmax()
            current_max = epsilon_series.max()
            
            # Update global maximum if needed
            if current_max > max_epsilon:
                max_epsilon = current_max
                selected_label_O = label_O
                selected_attribute = max_attr
                
        return selected_label_O, selected_attribute
            
    def step(self, X, transformed_df, O, selected_attribute, selected_label_O, changed_dict):
        """
        Perform a bias mitigation step
        
        Parameters:
        X: pandas.DataFrame
            Input dataset
        transformed_df: pandas.DataFrame
            Transformed dataset
        O: pandas.Series
            Protected attribute(s) to mitigate bias
        selected_attribute: str
            Attribute to be mitigated
        selected_label_O: str
            Protected attribute to be mitigated for the selected_attribute
        changed_dict: dict
            Dictionary of changes made to the dataset
        df_epsilon: dict, optional
            Precomputed epsilon values from Evaluator.calculate_epsilon
            If not provided, will be computed using Evaluator
        
        Returns:
        tuple
            (selected_label_O, selected_attribute) - The attribute pair with highest epsilon
        """
            
        # Update changed_dict based on attribute type
        if selected_attribute in self.cate_attrs:
            # Categorical attribute - perform rebin
            df_data_bm_temp = pd.concat([transformed_df.reset_index(drop=True), O.reset_index(drop=True)], axis=1)
            attr_bias = selected_attribute
            label_O = selected_label_O
            
            diff_change = []
            for i, j in combinations(df_data_bm_temp[label_O].unique(), 2):
                df_0 = df_data_bm_temp[[attr_bias, label_O]][df_data_bm_temp[label_O] == i].groupby(attr_bias).count()
                df_1 = df_data_bm_temp[[attr_bias, label_O]][df_data_bm_temp[label_O] == j].groupby(attr_bias).count()
                df_c = (df_1/(df_1.sum()) - df_0/(df_0.sum())).fillna(0).sort_values(by=label_O)
                diff = (df_c.max() + df_c.min()).values[0]
                change_temp = {df_c.index[-1]: df_c.index[0]}
                diff_change.append([change_temp, diff])
            
            diff_change = pd.DataFrame(diff_change, columns=['change', 'diff'])
            change = diff_change[diff_change['diff'].abs() == diff_change['diff'].abs().max()]['change'].values[0]
            
            # Update changed_dict
            if attr_bias not in changed_dict:
                changed_dict[attr_bias] = {}
            # Convert np.int64 to native Python int
            converted_change = {int(k): int(v) for k, v in change.items()}
            changed_dict[attr_bias].update(converted_change)
        else:
            # Numerical attribute - increment beta_O by 1
            attr_bias = selected_attribute
            if attr_bias not in changed_dict:
                change = {'beta_O': 0}
            else:
                change = changed_dict[attr_bias]
                change['beta_O'] = change.get('beta_O', 0) + 1
            
            if self.transformer.check_transform_validity(X, attr_bias, change, self.num_attrs, self.cate_attrs):
                changed_dict[attr_bias] = change
            else:
                changed_dict[attr_bias] = 'dropped'
            
        return changed_dict
    
    def mitigate(self, X, O, changed_dict=None):
        """
        Continuously perform bias mitigation steps based on PARAMS_STEP
        
        Parameters:
        X: pandas.DataFrame
            Input dataset
        O: pandas.Series
            Protected attribute(s) to mitigate bias
        changed_dict: dict, optional
            Initial dictionary of changes made to the dataset
            If not provided, will be initialized as empty dict
        
        Returns:
        tuple
            (transformed_df, final_changed_dict) - The transformed dataset and final changes
        """
        # Initialize changed_dict if not provided
        if changed_dict is None:
            changed_dict = {}
        
        # Create a copy of the dataframe to modify
        transformed_df = self.transformer.transform_data(X, changed_dict, self.num_attrs, self.cate_attrs)
        
        max_attempts = len(self.transformer.stream_data)  # Maximum number of attempts to decrease epsilon
        attempt = 0
        success = False
        
        # Create a temporary dictionary to track changes for each attempt
        temp_changed_dict = changed_dict.copy()
        temp_transformed_df = transformed_df.copy()
        
        initial_df_epsilon = self.evaluator.calculate_epsilon(temp_transformed_df, O, self.cate_attrs, self.num_attrs)
        selected_label_O, selected_attribute = self._find_max_epsilon_attribute(initial_df_epsilon)
        initial_max_epsilon = initial_df_epsilon[selected_label_O][selected_attribute]
        
        while attempt < max_attempts and not success:
            attempt += 1

            # Perform a mitigation step
            temp_changed_dict = self.step(X, temp_transformed_df, O, selected_attribute, selected_label_O, temp_changed_dict)
            if VERBOSE:
                print(f"Attempt {attempt}: Selected attribute = {selected_attribute}, Selected label_O = {selected_label_O}, Change = {temp_changed_dict[selected_attribute]}")
            
            # Apply the changes to the dataframe
            temp_transformed_df = self.transformer.transform_data(X, temp_changed_dict, self.num_attrs, self.cate_attrs)
            
            # Calculate new epsilon after change
            new_df_epsilon = self.evaluator.calculate_epsilon(temp_transformed_df, O, self.cate_attrs, self.num_attrs)
            new_max_epsilon = new_df_epsilon[selected_label_O][selected_attribute]
            
            if VERBOSE:
                print(f"Attempt {attempt}: After change, maximum epsilon for {selected_attribute} on {selected_label_O} = {new_max_epsilon:.6f}")
            
            # Check if epsilon decreased
            if new_max_epsilon < initial_max_epsilon:
                if VERBOSE:
                    print(f"Success! Epsilon decreased after attempt {attempt}.")
                # Update the main change dict and transformed df
                changed_dict.update(temp_changed_dict)
                transformed_df = temp_transformed_df
                success = True
            else:
                if VERBOSE:
                    print(f"Warning: Epsilon did not decrease in attempt {attempt}.")
                changed_dict[selected_attribute] = 'dropped'
        # After all attempts, update the main variables 

        return transformed_df, changed_dict