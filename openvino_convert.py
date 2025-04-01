import openvino as ov  # Updated to avoid deprecation warning
import torch
import torchvision.models as models

if __name__ == "__main__":
    # Load the DenseNet model architecture
    model = models.densenet161(weights=None)  # Initialize the model without pre-trained weights

    # Load the state dictionary into the model
    model_path = "densenet161.pth"  # Path to the saved model
    state_dict = torch.load(model_path)

    # Check if the state dictionary contains a "model_state_dict" key (common in TorchServe files)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore mismatched keys
    model.eval()  # Set the model to evaluation mode

    # Convert to OpenVINO format
    ov_model = ov.convert_model(
        model,
        example_input=torch.randn(1, 3, 224, 224)  # Provide an example input tensor
    )

    # Explicitly set names for the output tensors
    for i, output in enumerate(ov_model.outputs):
        output.get_tensor().set_names({f"output_{i}"})  # Use a set instead of a list

    # Save the OpenVINO model
    ov.serialize(ov_model, "openvino_model/model.xml", "openvino_model/model.bin")