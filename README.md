# cortex-scikit-onnx-boston-housing
This repository shows how to deploy a scikit model with [cortex](https://www.cortex.dev/) on AWS. The project only focuses on the deployment of the model. A simple *linear regression* was trained with [scikit-learn](https://scikit-learn.org/) framework on the [boston housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html). For more details take a look into [scikit-onnx-fastapi-example repository](https://github.com/naxty/scikit-onnx-fastapi-example). 


## 1. Scikit to ONNX conversion
Firstly install the following into your python environnment:
`pip install pandas sklearn sk2lonnx`

```
file= "model/linear_regression.joblib"
from joblib import load
model = load(file)

# Convert into ONNX format with onnxmltools
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('float_input', FloatTensorType([1, 13]))]
converted_onnx = convert_sklearn(model, initial_types=initial_type)
with open("model/linear_regression.onnx", "wb") as f:
    f.write(converted_onnx.SerializeToString())
```

## 2. cortex
We are deploying the [linear_regression.onnx model](model/) in an AWS enviornment. Cortex provides a workload to run ONNX models directly. 


### 2.1 Install
We are installing cortex directly in our AWS environment. But before we need to create a USER through IAM. We need to assign the `AdministratorAccess` policy to the USER since we need to create different resources such as S3 and EKS.

After we have set up the USER we can run the following script. I copied most parts of the installation scripts from [cortexlabs/cortex](https://github.com/cortexlabs/cortex#installation).
```
# Download
curl -O https://raw.githubusercontent.com/cortexlabs/cortex/0.8/cortex.sh
# Change permissions
chmod +x cortex.sh
# Install the Cortex CLI on your machine
./cortex.sh install cli
# Set AWS credentials
export AWS_ACCESS_KEY_ID=***
export AWS_SECRET_ACCESS_KEY=***
# Configure AWS instance settings (at least 4GB memory)
export CORTEX_NODE_TYPE=m5.large
export CORTEX_NODES_MIN=2
export CORTEX_NODES_MAX=2
export CORTEX_REGION=eu-west-2
# Provision infrastructure on AWS and install Cortex
./cortex.sh install
```

### 2.2 Develop and deploy
- Upload the [lienar_regression.onnx](model/linear_regression.onnx) to a S3 bucket `aws s3 cp model/linear_regression S3://BUCKET/boston_housting/linear_regression.onnx`. The cortex installation automatically creates a S3 bucket that we can use. 
- Implement `pre_inference`and `post_inference` in [boston_handler.py](boston_handler.py).
- Define deployment and API in [cortex.yaml](cortex.yaml) file.
- Run `cortex deploy`

### 2.3 Test the application
Run `cortex get linear-regression` to get the URL of the model. Edit the URL in `test.sh` and run the script. 