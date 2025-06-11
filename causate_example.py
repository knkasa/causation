# Create Causal graph.
# https://awadrahman.medium.com/causate-an-initiative-to-operationalize-causality-with-mlflow-9beaa9e4a1e8


from causate import CausalOpsEngine
import pandas as pd

data = pd.read_csv("dataset.csv")

cops = CausalOpsEngine(
    mode="create",
    algorithm="PC",
    algorithm_params={
        "variant": "original",
        "alpha": 0.9,
        "ci_test": "fisherz"
    },
    max_features=10
)

causal_matrix = cops.discover_causal_graph(data)

# creating causal graph.
cops.plot_causal_graph(
    causal_matrix_dataframe=causal_matrix,
    layout="random",
    layout_seed=51
)

# save model in log.
cops.log_causal_model()

# Run inference for new dataset using the pre-trained model.
cops_predict = CausalOpsEngine(mode="infer")

# Load the model from MLflow
logged_model = 'runs:/b58d1b3f31974c3bac602e8f53da3ece/model'
cops_predict.load_causal_model(logged_model)

# Check the model metadata
print(cops_predict.model_metadata)

data = pd.read_csv("dataset_inference.csv")
predicted_causal_matrix = cops_predict.infer_causal_model(data)

cops_predict.plot_causal_graph(
    causal_matrix_dataframe=predicted_causal_matrix,
    layout="random",
    layout_seed=51
)
