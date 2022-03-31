
import numpy as np

from external.mask_rcnn.mrcnn import visualize
import tritonclient.grpc as tritongrpcclient
import cv2
import os
from inferencing import saved_model_config
from inferencing.saved_model_preprocess import ForwardModel
model_config = saved_model_config.MY_INFERENCE_CONFIG
preprocess_obj = ForwardModel(model_config)
triton_client = tritongrpcclient.InferenceServerClient(url='#host url', verbose=True)
dir = r'# path of image dir'
imglist = os.listdir(dir)
for img in imglist:
    image_path = os.path.join(dir,img)
    image = cv2.imread(image_path)
    images = np.expand_dims(image, axis=0)
    molded_images, image_metas, windows = preprocess_obj.mold_inputs(images)
    molded_images = molded_images.astype(np.float32)
    image_metas = image_metas.astype(np.float32)
    # Validate image sizes
    # All images in a batch MUST be of the same size
    image_shape = molded_images[0].shape
    for g in molded_images[1:]:
        assert g.shape == image_shape, \
            "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

    # Anchors
    anchors = preprocess_obj.get_anchors(image_shape)
    anchors = np.broadcast_to(anchors, (images.shape[0],) + anchors.shape)
    np.expand_dims(image, axis=0)
    inputs = []
    outputs = []
    inputs.append(tritongrpcclient.InferInput(saved_model_config.INPUT_IMAGE, molded_images.shape, "FP32"))
    inputs.append(tritongrpcclient.InferInput(saved_model_config.INPUT_IMAGE_META, image_metas.shape, "FP32"))
    inputs.append(tritongrpcclient.InferInput(saved_model_config.INPUT_ANCHORS, anchors.shape, "FP32"))
    outputs.append(tritongrpcclient.InferRequestedOutput('mrcnn_detection/Reshape_1'))
    outputs.append(tritongrpcclient.InferRequestedOutput('mrcnn_mask/Reshape_1'))
    inputs[0].set_data_from_numpy(molded_images)
    inputs[1].set_data_from_numpy(image_metas)
    inputs[2].set_data_from_numpy(anchors)
    result = triton_client.infer(model_name='crop-mrcnn',
                                  inputs=inputs,
                                  outputs=outputs)
    output0_data = result.as_numpy('mrcnn_detection/Reshape_1')
    output1_data = result.as_numpy('mrcnn_mask/Reshape_1')
    result_dict = preprocess_obj.result_to_dict(images, molded_images, windows, result)[0]
    visualize.display_instances(
        image,
        result_dict['rois'],
        result_dict['mask'],
        result_dict['class'],
        ["BG", "rooftop"],
        result_dict['scores'],
        )