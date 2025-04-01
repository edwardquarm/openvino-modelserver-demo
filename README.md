# Serving a DenseNet Model with TorchServe and OpenVINO on KServe

This guide demonstrates how to serve a pre-trained DenseNet model using TorchServe and how to convert it directly to OpenVINO format for deployment on KServe.

## Steps

### 1. Download the Pre-trained Model
Download the DenseNet-161 model weights from PyTorch's model repository:
```bash
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
```

### 2. Prepare the Model Store
Create a directory to store the model artifacts:
```bash
mkdir model_store
```

### 3. Archive the Model for TorchServe
Use the `torch-model-archiver` tool to package the model into a `.mar` file:
```bash
torch-model-archiver \
  --model-name densenet161 \
  --version 1.0 \
  --model-file model.py \
  --serialized-file densenet161-8d451a50.pth \
  --export-path model_store \
  --extra-files index_to_name.json \
  --handler image_classifier \
  --config-file model-config.yaml \
  -f
```

### 4. Start TorchServe
Start the TorchServe server with the archived model:
```bash
torchserve --start --ncs --model-store model_store --models model_store/densenet161.mar --disable-token-auth --enable-model-api
```

### 5. Test the Model
Send a test image to the model for inference:
```bash
curl http://127.0.0.1:8080/predictions/densenet161 -T kitten.jpg
```

---

### 6. Convert the Model to OpenVINO
Convert the DenseNet model directly to OpenVINO format using OpenVINO's `convert_model` function:
```python
import torch
import torchvision
import openvino as ov

# Load the DenseNet model
model = torchvision.models.densenet161(weights='DEFAULT')

# Convert to OpenVINO format
ov_model = ov.convert_model(model)

# Save the OpenVINO model
ov.serialize(ov_model, "openvino_model/model.xml", "openvino_model/model.bin")
```

### 7. Organize the Model for KServe
Ensure the OpenVINO model files are organized in the following folder structure:
```
openvino_model/
└── 1/
    ├── model.xml
    ├── model.bin
```
Move the converted model files into the `1/` directory:
```bash
mkdir -p openvino_model/1
mv openvino_model/model.xml openvino_model/1/
mv openvino_model/model.bin openvino_model/1/
```

### 8. Run local Inference on the OpenVINO Model
Run the inference script to test the OpenVINO model:
```bash
python openvino_inference.py
```

### 9. Deploy the Model on KServe
Create a KServe InferenceService YAML file:
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: densenet-openvino
spec:
  predictor:
    model:
      modelFormat:
        name: openvino
      storageUri: "s3://your-bucket/openvino_model"
```

Apply the YAML file to your Kubernetes cluster:
```bash
kubectl apply -f inference-service.yaml
```

### 10. Test the Deployment
Send a test request to the deployed model:
```bash
curl -X POST http://<kserve-endpoint>/v1/models/densenet-openvino:predict -d '{"instances": [[...]]}'
```

## Notes
- Replace `model.py` with the actual script defining the DenseNet model architecture.
- Ensure `index_to_name.json` and `model-config.yaml` are correctly configured for your use case.
- Replace `<kserve-endpoint>` with your KServe endpoint.
- The test image (`kitten.jpg`) should be located in the specified path.

## References
- [TorchServe Documentation](https://pytorch.org/serve/)
- [DenseNet Model](https://pytorch.org/hub/pytorch_vision_densenet/)
- [OpenVINO Toolkit](https://docs.openvino.ai/latest/index.html)
- [KServe Documentation](https://kserve.github.io/)