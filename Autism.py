
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

try:
    from ucimlrepo import fetch_ucirepo
    UCI_AVAILABLE = True
except ImportError:
    UCI_AVAILABLE = False
    print("Note: ucimlrepo not installed. Install with: pip install ucimlrepo")

class EnhancedAutismDetector:
    """
    Enhanced Autism Detection System using Machine Learning
    Supports both synthetic and real datasets
    """

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = ""
        self.feature_names = []

    def load_real_dataset(self, dataset_type='adult'):
        """
        Load real autism dataset from UCI ML Repository

        Parameters:
        dataset_type: 'adult', 'children', or 'adolescent'
        """
        if not UCI_AVAILABLE:
            print("UCI ML Repository not available. Using synthetic data instead.")
            return self.create_sample_dataset()

        try:
            if dataset_type == 'adult':
                # UCI dataset ID 426 - Autism Screening Adult
                autism_data = fetch_ucirepo(id=426)
            elif dataset_type == 'children':
                # UCI dataset ID 419 - Autism Screening Children  
                autism_data = fetch_ucirepo(id=419)
            elif dataset_type == 'adolescent':
                # UCI dataset ID 420 - Autism Screening Adolescent
                autism_data = fetch_ucirepo(id=420)
            else:
                raise ValueError("dataset_type must be 'adult', 'children', or 'adolescent'")

            # Combine features and targets
            X = autism_data.data.features
            y = autism_data.data.targets

            # Combine into single DataFrame
            df = pd.concat([X, y], axis=1)

            print(f"Real {dataset_type} autism dataset loaded successfully!")
            print(f"Dataset shape: {df.shape}")
            print(f"Features: {list(X.columns)}")
            print(f"Target: {list(y.columns)}")

            return df

        except Exception as e:
            print(f"Error loading real dataset: {e}")
            print("Using synthetic data instead.")
            return self.create_sample_dataset()

    def create_sample_dataset(self, n_samples=500):
        """
        Create a sample dataset similar to UCI autism dataset
        """
        np.random.seed(42)

        # Generate Q-CHAT-10 screening scores
        data = {
            'A1_Score': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'A2_Score': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'A3_Score': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'A4_Score': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'A5_Score': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'A6_Score': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'A7_Score': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'A8_Score': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
            'A9_Score': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'A10_Score': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),

            # Demographic information
            'age': np.random.randint(18, 65, n_samples),
            'gender': np.random.choice(['m', 'f'], n_samples),
            'ethnicity': np.random.choice(['White-European', 'Latino', 'Asian', 'Black', 'Others'], n_samples),
            'jaundice': np.random.choice(['yes', 'no'], n_samples, p=[0.1, 0.9]),
            'autism_family_history': np.random.choice(['yes', 'no'], n_samples, p=[0.15, 0.85]),
            'country_of_res': np.random.choice(['United States', 'Brazil', 'Spain', 'Egypt', 'Others'], n_samples),
            'used_app_before': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
            'screening_type': np.random.choice([1, 2, 3], n_samples),
            'relation': np.random.choice(['Self', 'Parent', 'Relative', 'Health care professional'], n_samples)
        }

        # Create target variable based on screening scores
        screening_scores = sum(data[f'A{i}_Score'] for i in range(1, 11))
        data['Class/ASD'] = np.where(screening_scores >= 6, 1, 0)

        # Add some noise to make it more realistic
        noise = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        data['Class/ASD'] = np.where(noise == 1, 1 - data['Class/ASD'], data['Class/ASD'])

        return pd.DataFrame(data)

    def preprocess_data(self, df):
        """
        Preprocess the data for machine learning
        """
        df_processed = df.copy()

        # Handle different target column names
        target_cols = ['Class/ASD', 'Class_ASD', 'autism', 'ASD']
        target_col = None
        for col in target_cols:
            if col in df_processed.columns:
                target_col = col
                break

        if target_col is None:
            # If no target column found, create one based on screening scores
            screening_cols = [f'A{i}_Score' for i in range(1, 11)]
            available_cols = [col for col in screening_cols if col in df_processed.columns]
            if available_cols:
                screening_scores = df_processed[available_cols].sum(axis=1)
                df_processed['Class/ASD'] = np.where(screening_scores >= 6, 1, 0)
        elif target_col != 'Class/ASD':
            df_processed['Class/ASD'] = df_processed[target_col]

        # Replace text values with binary encoding
        df_processed = df_processed.replace({'yes': 1, 'no': 0, 'YES': 1, 'NO': 0})

        # Encode categorical variables
        le = LabelEncoder()
        categorical_columns = ['gender', 'ethnicity', 'country_of_res', 'relation', 'contry_of_res']

        for col in categorical_columns:
            if col in df_processed.columns:
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))

        # Feature engineering - create total score
        screening_cols = [f'A{i}_Score' for i in range(1, 11)]
        available_cols = [col for col in screening_cols if col in df_processed.columns]
        if available_cols:
            df_processed['total_score'] = df_processed[available_cols].sum(axis=1)

        return df_processed

    def train_models(self, X_train, y_train):
        """
        Train multiple machine learning models
        """
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        }

        results = {}

        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            results[name] = model

        return results

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        """
        results = {}

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            print(f"\n{name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC Score: {auc_score:.4f}")
            print("  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['No ASD', 'ASD'], zero_division=0))

        # Select best model
        self.best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        self.best_model = self.models[self.best_model_name]

        print(f"\nBest performing model: {self.best_model_name}")
        print(f"Best AUC Score: {results[self.best_model_name]['auc_score']:.4f}")

        return results

    def predict_autism(self, features_dict):
        """
        Predict autism likelihood for a new case
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")

        # Convert features to DataFrame
        features_df = pd.DataFrame([features_dict])

        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(features_df.columns)
        for feature in missing_features:
            features_df[feature] = 0  # Default value

        # Reorder columns to match training data
        features_df = features_df[self.feature_names]

        # Scale features
        features_scaled = self.scaler.transform(features_df)

        # Make prediction
        prediction = self.best_model.predict(features_scaled)[0]
        probability = self.best_model.predict_proba(features_scaled)[0, 1]

        return prediction, probability

    def q_chat_questions(self):
        """
        Display Q-CHAT-10 questions for reference
        """
        questions = [
            "Q1: Does your child look at you when you call their name?",
            "Q2: How easy is it for you to get eye contact with your child?",
            "Q3: Does your child point to indicate that they want something?",
            "Q4: Does your child point to share interest with you?",
            "Q5: Does your child pretend during play?",
            "Q6: Does your child follow where you're pointing?",
            "Q7: If you turn around to look at something, does your child look to see what you are looking at?",
            "Q8: Does your child try to attract your attention to their own activity?",
            "Q9: Does your child cuddle when you try to cuddle them?",
            "Q10: Does your child respond to their name when called?"
        ]

        print("Q-CHAT-10 Screening Questions:")
        print("=" * 50)
        for i, question in enumerate(questions, 1):
            print(f"{question}")
        print("\nScoring: 0 = Often/Always, 1 = Sometimes/Rarely/Never")
        print("Higher scores indicate higher likelihood of ASD traits")

    def run_complete_analysis(self, use_real_data=True, dataset_type='adult'):
        """
        Run the complete autism detection analysis
        """
        print("Enhanced Autism Detection Program")
        print("=" * 50)

        # Load dataset
        if use_real_data:
            print(f"Loading real {dataset_type} autism dataset...")
            df = self.load_real_dataset(dataset_type)
        else:
            print("Creating synthetic dataset...")
            df = self.create_sample_dataset()

        print(f"Dataset loaded with {len(df)} samples")
        print(f"ASD prevalence: {df['Class/ASD'].mean():.2%}")

        # Preprocess data
        print("\nPreprocessing data...")
        df_processed = self.preprocess_data(df)

        # Prepare features and target
        X = df_processed.drop(['Class/ASD'], axis=1)
        y = df_processed['Class/ASD']
        self.feature_names = list(X.columns)

        print(f"Features: {len(self.feature_names)}")
        print(f"Samples: {len(y)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Train models
        print("\nTraining models...")
        self.train_models(X_train_scaled, y_train)

        # Evaluate models
        print("\nEvaluating models...")
        results = self.evaluate_models(X_test_scaled, y_test)

        return results

    def interactive_screening(self):
        """
        Interactive screening interface
        """
        print("\n" + "=" * 50)
        print("INTERACTIVE AUTISM SCREENING")
        print("=" * 50)

        # Display questions
        self.q_chat_questions()

        print("\n" + "=" * 50)
        print("EXAMPLE SCREENING CASE")
        print("=" * 50)

        # Example high-risk case
        high_risk_case = {
            'A1_Score': 1, 'A2_Score': 1, 'A3_Score': 1, 'A4_Score': 1, 'A5_Score': 1,
            'A6_Score': 1, 'A7_Score': 1, 'A8_Score': 0, 'A9_Score': 1, 'A10_Score': 1,
            'age': 25, 'gender': 1, 'ethnicity': 0, 'jaundice': 0,
            'autism_family_history': 1, 'country_of_res': 0, 'used_app_before': 0,
            'screening_type': 1, 'relation': 0, 'total_score': 8
        }

        print("High-Risk Example:")
        print("Q-CHAT-10 Scores:", [high_risk_case[f'A{i}_Score'] for i in range(1, 11)])
        print(f"Total Score: {high_risk_case['total_score']}/10")
        print(f"Age: {high_risk_case['age']} years")
        print(f"Family History: {'Yes' if high_risk_case['autism_family_history'] else 'No'}")

        # Make prediction
        prediction, probability = self.predict_autism(high_risk_case)

        print(f"\nSCREENING RESULTS:")
        print(f"Prediction: {'ASD Traits Likely' if prediction == 1 else 'ASD Traits Unlikely'}")
        print(f"Probability: {probability:.1%}")
        print(f"Risk Level: {'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.3 else 'LOW'}")

        if probability > 0.5:
            print("\nRECOMMENDATION: Consider professional evaluation")
        else:
            print("\nRECOMMENDATION: Screening suggests low likelihood of ASD")

        # Low-risk example
        print("\n" + "-" * 30)
        print("Low-Risk Example:")

        low_risk_case = {
            'A1_Score': 0, 'A2_Score': 0, 'A3_Score': 0, 'A4_Score': 0, 'A5_Score': 0,
            'A6_Score': 0, 'A7_Score': 0, 'A8_Score': 1, 'A9_Score': 0, 'A10_Score': 0,
            'age': 30, 'gender': 0, 'ethnicity': 1, 'jaundice': 0,
            'autism_family_history': 0, 'country_of_res': 1, 'used_app_before': 0,
            'screening_type': 1, 'relation': 0, 'total_score': 1
        }

        print("Q-CHAT-10 Scores:", [low_risk_case[f'A{i}_Score'] for i in range(1, 11)])
        print(f"Total Score: {low_risk_case['total_score']}/10")

        prediction2, probability2 = self.predict_autism(low_risk_case)

        print(f"\nSCREENING RESULTS:")
        print(f"Prediction: {'ASD Traits Likely' if prediction2 == 1 else 'ASD Traits Unlikely'}")
        print(f"Probability: {probability2:.1%}")
        print(f"Risk Level: {'HIGH' if probability2 > 0.7 else 'MEDIUM' if probability2 > 0.3 else 'LOW'}")

        return (prediction, probability), (prediction2, probability2)

# Main execution function
def main():
    """
    Main function to run the autism detection program
    """
    print("Enhanced Autism Detection Program")
    print("Using Machine Learning for Early Screening")
    print("=" * 50)

    # Create detector instance
    detector = EnhancedAutismDetector()

    # Run complete analysis with real data
    try:
        results = detector.run_complete_analysis(use_real_data=True, dataset_type='adult')
    except Exception as e:
        print(f"Error with real data: {e}")
        print("Falling back to synthetic data...")
        results = detector.run_complete_analysis(use_real_data=False)

    # Run interactive screening
    detector.interactive_screening()



if __name__ == "__main__":
    main()
