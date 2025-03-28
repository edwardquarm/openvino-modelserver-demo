import torchvision
import openvino as ov

if __name__ == "__main__":

    # Load the DenseNet model
    model = torchvision.models.densenet161(weights='DEFAULT')

    # Convert to OpenVINO format
    ov_model = ov.convert_model(model)

    # Save the OpenVINO model
    ov.serialize(ov_model, "openvino_model/model.xml", "openvino_model/model.bin")