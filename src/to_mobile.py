import torch
import torchvision
import model as model
from mobilenetv2 import MobileNetV2

keypointsNumber = 14

# model_dir = 'mobile/resnet18.pth'
# mobile_dir = './mobile/resnet18.pt'
model_dir = 'mobile/mobilenetv2.pth'
mobile_dir = './mobile/mobilenetv2.pt'

model = model.A2J_model(num_classes=keypointsNumber)
model.load_state_dict(torch.load(model_dir))
model.eval()
example = torch.rand(1, 1, 176, 176)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save(mobile_dir)
