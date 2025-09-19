import dill
from sklearn.ensemble import VotingRegressor

# Define the path to your saved model
MODEL_FILE_PATH = "artifacts/model.pkl"

def inspect_saved_model(file_path):
    """Loads and inspects the saved model file."""
    try:
        with open(file_path, "rb") as f:
            model = dill.load(f)
        
        print(f"✅ Successfully loaded model from: {file_path}\n")

        # Check if the model is a VotingRegressor ensemble
        if isinstance(model, VotingRegressor):
            print("The saved model is a 'VotingRegressor' ensemble. 🤖")
            
            # Print the base models that make up the ensemble
            print("\nIt is made up of the following base models:")
            for name, estimator in model.estimators_:
                print(f"  - {name}: {estimator.__class__.__name__}")
        
        # If it's not an ensemble, it's a single model
        else:
            model_name = model.__class__.__name__
            print(f"The saved model is a single model: '{model_name}' 🥇")

    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_saved_model(MODEL_FILE_PATH)