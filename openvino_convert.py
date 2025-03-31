import torchvision
import openvino as ov
import torch

if __name__ == "__main__":

    # Load the DenseNet model
    model = torchvision.models.densenet161(weights='DEFAULT')

    # Convert to OpenVINO format
    ov_model = ov.convert_model(model, example_input=torch.randn(1, 3, 224, 224))

    # Save the OpenVINO model
    ov.serialize(ov_model, "openvino_model/model.xml", "openvino_model/model.bin")