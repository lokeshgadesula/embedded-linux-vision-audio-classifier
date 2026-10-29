class TFLiteModel:
 def __init__(self,path):
  try:from tensorflow.lite import Interpreter
  except ImportError:from tflite_runtime.interpreter import Interpreter
  self.i=Interpreter(model_path=path,num_threads=2);self.i.allocate_tensors()
 def __call__(self,x):
  a=self.i.get_input_details()[0];b=self.i.get_output_details()[0];self.i.set_tensor(a["index"],x.astype(a["dtype"]));self.i.invoke();return self.i.get_tensor(b["index"])[0]
def opencv_frame(index=0):
 import cv2;c=cv2.VideoCapture(index);ok,f=c.read();c.release()
 if not ok:raise RuntimeError("camera")
 return f
def pytorch_model(classes=2):
 import torch;return torch.nn.Sequential(torch.nn.Flatten(),torch.nn.LazyLinear(16),torch.nn.ReLU(),torch.nn.Linear(16,classes))
