# Triton-Inference-Serving-For-Maskrcnn
**Triton Inference Server**  

Triton Inference Server streamlines AI inference by enabling teams to deploy, run and scale trained AI models from any framework on any GPU- or CPU-based infrastructure. It provides AI researchers and data scientists the freedom to choose the right framework for their projects without impacting production deployment. It also helps developers deliver high-performance inference across cloud, on-prem, edge, and embedded devices.  

**Mask RCNN**  

Mask R-CNN, or Mask RCNN, is a Convolutional Neural Network (CNN) and state-of-the-art in terms of image segmentation and instance segmentation. Mask R-CNN was developed on top of Faster R-CNN, a Region-Based Convolutional Neural Network.  

**Converting mask_rcnn_coco.h5 weight file to tensorrt weight file format**  

For converting the weight file I had changed some parameters in the maskrcnn '**model.py**' and you can use '**tensorrtt.py**' for conversions. After converting the weight file along with the '**config.pbtxt**' file need to upload it in the server where u hosted triton serving.  

**Inferencing**  

For inferencing u can use '**infrencing_triton_serving.py**' file.

**Reference**

-https://github.com/triton-inference-server/server  

-https://github.com/NVIDIA/TensorRT/tree/master/samples/sampleUffMaskRCNN  

-https://github.com/matterport/Mask_RCNN

