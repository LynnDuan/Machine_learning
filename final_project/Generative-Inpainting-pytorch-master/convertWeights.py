import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

def convertWeights(tf_model_path, pytorch_model_path):
    # tf
    reader = pywrap_tensorflow.NewCheckpointReader(tf_model_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # pytorch
    pre_gen_dict = torch.load(pytorch_model_path)
    # pre_dis_dict = torch.load(pytorch_model_path)

    # convert tf to torch
    # stage_1 conv downsample
    w_conv1_kernel = reader.get_tensor("inpaint_net/conv1/kernel")
    pre_gen_dict["stage_1.down.out.0.conv.0.weight"] = torch.tensor(np.transpose(w_conv1_kernel, [3, 2, 1, 0])).double()
    w_conv1_bias = reader.get_tensor("inpaint_net/conv1/bias")
    pre_gen_dict["stage_1.down.out.0.conv.0.bias"] = torch.tensor(w_conv1_bias) 

    w_conv2_kernel = reader.get_tensor("inpaint_net/conv2_downsample/kernel")
    pre_gen_dict["stage_1.down.out.1.conv.0.weight"] = torch.tensor(np.transpose(w_conv2_kernel, [3, 2, 1, 0])).double()
    w_conv2_bias = reader.get_tensor("inpaint_net/conv2_downsample/bias")
    pre_gen_dict["stage_1.down.out.1.conv.0.bias"] = torch.tensor(w_conv2_bias) 

    w_conv3_kernel = reader.get_tensor("inpaint_net/conv3/kernel")
    pre_gen_dict["stage_1.down.out.2.conv.0.weight"] = torch.tensor(np.transpose(w_conv3_kernel, [3, 2, 1, 0])).double()
    w_conv3_bias = reader.get_tensor("inpaint_net/conv3/bias")
    pre_gen_dict["stage_1.down.out.2.conv.0.bias"] = torch.tensor(w_conv3_bias) 

    w_conv4_kernel = reader.get_tensor("inpaint_net/conv4_downsample/kernel")
    pre_gen_dict["stage_1.down.out.3.conv.0.weight"] = torch.tensor(np.transpose(w_conv4_kernel, [3, 2, 1, 0])).double()
    w_conv4_bias = reader.get_tensor("inpaint_net/conv4_downsample/bias")
    pre_gen_dict["stage_1.down.out.3.conv.0.bias"] = torch.tensor(w_conv4_bias) 

    w_conv5_kernel = reader.get_tensor("inpaint_net/conv5/kernel")
    pre_gen_dict["stage_1.down.out.4.conv.0.weight"] = torch.tensor(np.transpose(w_conv5_kernel, [3, 2, 1, 0])).double()
    w_conv5_bias = reader.get_tensor("inpaint_net/conv5/bias")
    pre_gen_dict["stage_1.down.out.4.conv.0.bias"] = torch.tensor(w_conv5_bias) 

    w_conv6_kernel = reader.get_tensor("inpaint_net/conv6/kernel")
    pre_gen_dict["stage_1.down.out.5.conv.0.weight"] = torch.tensor(np.transpose(w_conv6_kernel, [3, 2, 1, 0])).double()
    w_conv6_bias = reader.get_tensor("inpaint_net/conv6/bias")
    pre_gen_dict["stage_1.down.out.5.conv.0.bias"] = torch.tensor(w_conv6_bias) 

    print("finish stage_1 conv downsample")

    # stage_1 dilation
    w_conv7_atrous_kernel = reader.get_tensor("inpaint_net/conv7_atrous/kernel")
    pre_gen_dict["stage_1.atrous.out.0.conv.0.weight"] = torch.tensor(np.transpose(w_conv7_atrous_kernel, [3, 2, 1, 0])).double()
    w_conv7_atrous_bias = reader.get_tensor("inpaint_net/conv7_atrous/bias")
    pre_gen_dict["stage_1.atrous.out.0.conv.0.bias"] = torch.tensor(w_conv7_atrous_bias) 

    w_conv8_atrous_kernel = reader.get_tensor("inpaint_net/conv8_atrous/kernel")
    pre_gen_dict["stage_1.atrous.out.0.conv.0.weight"] = torch.tensor(np.transpose(w_conv8_atrous_kernel, [3, 2, 1, 0])).double()
    w_conv8_atrous_bias = reader.get_tensor("inpaint_net/conv8_atrous/bias")
    pre_gen_dict["stage_1.atrous.out.0.conv.0.bias"] = torch.tensor(w_conv8_atrous_bias) 

    w_conv9_atrous_kernel = reader.get_tensor("inpaint_net/conv9_atrous/kernel")
    pre_gen_dict["stage_1.atrous.out.0.conv.0.weight"] = torch.tensor(np.transpose(w_conv9_atrous_kernel, [3, 2, 1, 0])).double()
    w_conv9_atrous_bias = reader.get_tensor("inpaint_net/conv9_atrous/bias")
    pre_gen_dict["stage_1.atrous.out.0.conv.0.bias"] = torch.tensor(w_conv9_atrous_bias) 

    w_conv10_atrous_kernel = reader.get_tensor("inpaint_net/conv10_atrous/kernel")
    pre_gen_dict["stage_1.atrous.out.0.conv.0.weight"] = torch.tensor(np.transpose(w_conv10_atrous_kernel, [3, 2, 1, 0])).double()
    w_conv10_atrous_bias = reader.get_tensor("inpaint_net/conv10_atrous/bias")
    pre_gen_dict["stage_1.atrous.out.0.conv.0.bias"] = torch.tensor(w_conv10_atrous_bias) 

    print("finish stage_1 dilation")

    # stage_1 conv upsample
    w_conv11_kernel = reader.get_tensor("inpaint_net/conv11/kernel")
    pre_gen_dict["stage_1.up.out.0.conv.0.weight"] = torch.tensor(np.transpose(w_conv11_kernel, [3, 2, 1, 0])).double()
    w_conv11_bias = reader.get_tensor("inpaint_net/conv11/bias")
    pre_gen_dict["stage_1.up.out.0.conv.0.bias"] = torch.tensor(w_conv11_bias) 

    w_conv12_kernel = reader.get_tensor("inpaint_net/conv12/kernel")
    pre_gen_dict["stage_1.up.out.1.conv.0.weight"] = torch.tensor(np.transpose(w_conv12_kernel, [3, 2, 1, 0])).double()
    w_conv12_bias = reader.get_tensor("inpaint_net/conv12/bias")
    pre_gen_dict["stage_1.up.out.1.conv.0.bias"] = torch.tensor(w_conv12_bias) 

    w_conv13_kernel = reader.get_tensor("inpaint_net/conv13_upsample/conv13_upsample_conv/kernel")
    pre_gen_dict["stage_1.up.out.3.conv.0.weight"] = torch.tensor(np.transpose(w_conv13_kernel, [3, 2, 1, 0])).double()
    w_conv13_bias = reader.get_tensor("inpaint_net/conv13_upsample/conv13_upsample_conv/bias")
    pre_gen_dict["stage_1.up.out.3.conv.0.bias"] = torch.tensor(w_conv13_bias) 

    w_conv14_kernel = reader.get_tensor("inpaint_net/conv14/kernel")
    pre_gen_dict["stage_1.up.out.4.conv.0.weight"] = torch.tensor(np.transpose(w_conv14_kernel, [3, 2, 1, 0])).double()
    w_conv14_bias = reader.get_tensor("inpaint_net/conv14/bias")
    pre_gen_dict["stage_1.up.out.4.conv.0.bias"] = torch.tensor(w_conv14_bias) 

    w_conv15_kernel = reader.get_tensor("inpaint_net/conv15_upsample/conv15_upsample_conv/kernel")
    pre_gen_dict["stage_1.up.out.6.conv.0.weight"] = torch.tensor(np.transpose(w_conv15_kernel, [3, 2, 1, 0])).double()
    w_conv15_bias = reader.get_tensor("inpaint_net/conv15_upsample/conv15_upsample_conv/bias")
    pre_gen_dict["stage_1.up.out.6.conv.0.bias"] = torch.tensor(w_conv15_bias) 

    w_conv16_kernel = reader.get_tensor("inpaint_net/conv16/kernel")
    pre_gen_dict["stage_1.up.out.7.conv.0.weight"] = torch.tensor(np.transpose(w_conv16_kernel, [3, 2, 1, 0])).double()
    w_conv16_bias = reader.get_tensor("inpaint_net/conv16/bias")
    pre_gen_dict["stage_1.up.out.7.conv.0.bias"] = torch.tensor(w_conv16_bias) 

    w_conv17_kernel = reader.get_tensor("inpaint_net/conv17/kernel")
    pre_gen_dict["stage_1.up.out.8.conv.0.weight"] = torch.tensor(np.transpose(w_conv17_kernel, [3, 2, 1, 0])).double()
    w_conv17_bias = reader.get_tensor("inpaint_net/conv17/bias")
    pre_gen_dict["stage_1.up.out.8.conv.0.bias"] = torch.tensor(w_conv17_bias) 

    print("finish stage_1 conv upsample")

    # stage_2 conv downsample
    w_xconv1_kernel = reader.get_tensor("inpaint_net/xconv1/kernel")
    pre_gen_dict["stage_2.down_conv_branch.out.0.conv.0.weight"] = torch.tensor(np.transpose(w_xconv1_kernel, [3, 2, 1, 0])).double()
    w_xconv1_bias = reader.get_tensor("inpaint_net/xconv1/bias")
    pre_gen_dict["stage_2.down_conv_branch.out.0.conv.0.bias"] = torch.tensor(w_xconv1_bias) 

    # pytorch net different from tensorflow ----- start
    w_xconv2_kernel = reader.get_tensor("inpaint_net/xconv3/kernel")
    pre_gen_dict["stage_2.down_conv_branch.out.1.conv.0.weight"] = torch.tensor(np.transpose(w_xconv2_kernel, [3, 2, 1, 0])).double()
    w_xconv2_bias = reader.get_tensor("inpaint_net/xconv3/bias")
    pre_gen_dict["stage_2.down_conv_branch.out.1.conv.0.bias"] = torch.tensor(w_xconv2_bias) 

    w_xconv3_kernel = reader.get_tensor("inpaint_net/xconv4_downsample/kernel")
    pre_gen_dict["stage_2.down_conv_branch.out.2.conv.0.weight"] = torch.tensor(np.transpose(w_xconv3_kernel, [3, 2, 1, 0])).double()
    w_xconv3_bias = reader.get_tensor("inpaint_net/xconv4_downsample/bias")
    pre_gen_dict["stage_2.down_conv_branch.out.2.conv.0.bias"] = torch.tensor(w_xconv3_bias) 

    w_xconv4_kernel = reader.get_tensor("inpaint_net/xconv5/kernel")
    pre_gen_dict["stage_2.down_conv_branch.out.3.conv.0.weight"] = torch.tensor(np.transpose(w_xconv4_kernel, [3, 2, 1, 0])).double()
    w_xconv4_bias = reader.get_tensor("inpaint_net/xconv5/bias")
    pre_gen_dict["stage_2.down_conv_branch.out.3.conv.0.bias"] = torch.tensor(w_xconv4_bias) 

    w_xconv5_kernel = reader.get_tensor("inpaint_net/xconv6/kernel")
    pre_gen_dict["stage_2.down_conv_branch.out.4.conv.0.weight"] = torch.tensor(np.transpose(w_xconv5_kernel, [3, 2, 1, 0])).double()
    w_xconv5_bias = reader.get_tensor("inpaint_net/xconv6/bias")
    pre_gen_dict["stage_2.down_conv_branch.out.4.conv.0.bias"] = torch.tensor(w_xconv5_bias) 

    print("finish stage_2 conv downsample")

    # stage_2 attn downsample
    w_pmconv1_kernel = reader.get_tensor("inpaint_net/pmconv1/kernel")
    pre_gen_dict["stage_2.down_attn_branch.out.0.conv.0.weight"] = torch.tensor(np.transpose(w_pmconv1_kernel, [3, 2, 1, 0])).double()
    w_pmconv1_bias = reader.get_tensor("inpaint_net/pmconv1/bias")
    pre_gen_dict["stage_2.down_attn_branch.out.0.conv.0.bias"] = torch.tensor(w_pmconv1_bias) 

    w_pmconv2_kernel = reader.get_tensor("inpaint_net/pmconv3/kernel")
    pre_gen_dict["stage_2.down_attn_branch.out.1.conv.0.weight"] = torch.tensor(np.transpose(w_pmconv2_kernel, [3, 2, 1, 0])).double()
    w_pmconv2_bias = reader.get_tensor("inpaint_net/pmconv3/bias")
    pre_gen_dict["stage_2.down_attn_branch.out.1.conv.0.bias"] = torch.tensor(w_pmconv2_bias) 

    # pytorch net different from tensorflow ----- end

    w_pmconv4_kernel = reader.get_tensor("inpaint_net/pmconv4_downsample/kernel")
    pre_gen_dict["stage_2.down_attn_branch.out.3.conv.0.weight"] = torch.tensor(np.transpose(w_pmconv4_kernel, [3, 2, 1, 0])).double()
    w_pmconv4_bias = reader.get_tensor("inpaint_net/pmconv4_downsample/bias")
    pre_gen_dict["stage_2.down_attn_branch.out.3.conv.0.bias"] = torch.tensor(w_pmconv4_bias) 

    w_pmconv5_kernel = reader.get_tensor("inpaint_net/pmconv5/kernel")
    pre_gen_dict["stage_2.down_attn_branch.out.4.conv.0.weight"] = torch.tensor(np.transpose(w_pmconv5_kernel, [3, 2, 1, 0])).double()
    w_pmconv5_bias = reader.get_tensor("inpaint_net/pmconv5/bias")
    pre_gen_dict["stage_2.down_attn_branch.out.4.conv.0.bias"] = torch.tensor(w_pmconv5_bias) 

    w_pmconv6_kernel = reader.get_tensor("inpaint_net/pmconv6/kernel")
    pre_gen_dict["stage_2.down_attn_branch.out.5.conv.0.weight"] = torch.tensor(np.transpose(w_pmconv6_kernel, [3, 2, 1, 0])).double()
    w_pmconv6_bias = reader.get_tensor("inpaint_net/pmconv6/bias")
    pre_gen_dict["stage_2.down_attn_branch.out.5.conv.0.bias"] = torch.tensor(w_pmconv6_bias) 

    print("finish stage_2 attn downsample")

    # stage_2 dilation
    w_xconv7_atrous_kernel = reader.get_tensor("inpaint_net/xconv7_atrous/kernel")
    pre_gen_dict["stage_2.atrous.out.0.conv.0.weight"] = torch.tensor(np.transpose(w_xconv7_atrous_kernel, [3, 2, 1, 0])).double()
    w_xconv7_atrous_bias = reader.get_tensor("inpaint_net/xconv7_atrous/bias")
    pre_gen_dict["stage_2.atrous.out.0.conv.0.bias"] = torch.tensor(w_xconv7_atrous_bias) 

    w_xconv8_atrous_kernel = reader.get_tensor("inpaint_net/xconv8_atrous/kernel")
    pre_gen_dict["stage_2.atrous.out.1.conv.0.weight"] = torch.tensor(np.transpose(w_xconv8_atrous_kernel, [3, 2, 1, 0])).double()
    w_xconv8_atrous_bias = reader.get_tensor("inpaint_net/xconv8_atrous/bias")
    pre_gen_dict["stage_2.atrous.out.1.conv.0.bias"] = torch.tensor(w_xconv8_atrous_bias) 

    w_xconv9_atrous_kernel = reader.get_tensor("inpaint_net/xconv9_atrous/kernel")
    pre_gen_dict["stage_2.atrous.out.2.conv.0.weight"] = torch.tensor(np.transpose(w_xconv9_atrous_kernel, [3, 2, 1, 0])).double()
    w_xconv9_atrous_bias = reader.get_tensor("inpaint_net/xconv9_atrous/bias")
    pre_gen_dict["stage_2.atrous.out.2.conv.0.bias"] = torch.tensor(w_xconv9_atrous_bias) 

    w_xconv10_atrous_kernel = reader.get_tensor("inpaint_net/xconv10_atrous/kernel")
    pre_gen_dict["stage_2.atrous.out.3.conv.0.weight"] = torch.tensor(np.transpose(w_xconv10_atrous_kernel, [3, 2, 1, 0])).double()
    w_xconv10_atrous_bias = reader.get_tensor("inpaint_net/xconv10_atrous/bias")
    pre_gen_dict["stage_2.atrous.out.3.conv.0.bias"] = torch.tensor(w_xconv10_atrous_bias) 
    
    print("finish stage_2 dilation")

    # stage_2 CAttn
    w_pmconv9_kernel = reader.get_tensor("inpaint_net/pmconv9/kernel")
    pre_gen_dict["stage_2.CAttn.out.0.conv.0.weight"] = torch.tensor(np.transpose(w_pmconv9_kernel, [3, 2, 1, 0])).double()
    w_pmconv9_bias = reader.get_tensor("inpaint_net/pmconv9/bias")
    pre_gen_dict["stage_2.CAttn.out.0.conv.0.bias"] = torch.tensor(w_pmconv9_bias) 

    w_pmconv10_kernel = reader.get_tensor("inpaint_net/pmconv10/kernel")
    pre_gen_dict["stage_2.CAttn.out.1.conv.0.weight"] = torch.tensor(np.transpose(w_pmconv10_kernel, [3, 2, 1, 0])).double()
    w_pmconv10_bias = reader.get_tensor("inpaint_net/pmconv10/bias")
    pre_gen_dict["stage_2.CAttn.out.1.conv.0.bias"] = torch.tensor(w_pmconv10_bias) 

    print("finish stage_2 CAttn")

    # stage_2 upsample
    w_allconv11_kernel = reader.get_tensor("inpaint_net/allconv11/kernel")
    pre_gen_dict["stage_2.up.out.0.conv.0.weight"] = torch.tensor(np.transpose(w_allconv11_kernel, [3, 2, 1, 0])).double()
    w_allconv11_bias = reader.get_tensor("inpaint_net/allconv11/bias")
    pre_gen_dict["stage_2.up.out.0.conv.0.bias"] = torch.tensor(w_allconv11_bias) 

    w_allconv12_kernel = reader.get_tensor("inpaint_net/allconv12/kernel")
    pre_gen_dict["stage_2.up.out.1.conv.0.weight"] = torch.tensor(np.transpose(w_allconv12_kernel, [3, 2, 1, 0])).double()
    w_allconv12_bias = reader.get_tensor("inpaint_net/allconv12/bias")
    pre_gen_dict["stage_2.up.out.1.conv.0.bias"] = torch.tensor(w_allconv12_bias) 

    w_allconv13_kernel = reader.get_tensor("inpaint_net/allconv13_upsample/allconv13_upsample_conv/kernel")
    pre_gen_dict["stage_2.up.out.3.conv.0.weight"] = torch.tensor(np.transpose(w_allconv13_kernel, [3, 2, 1, 0])).double()
    w_allconv13_bias = reader.get_tensor("inpaint_net/allconv13_upsample/allconv13_upsample_conv/bias")
    pre_gen_dict["stage_2.up.out.3.conv.0.bias"] = torch.tensor(w_allconv13_bias) 

    w_allconv14_kernel = reader.get_tensor("inpaint_net/allconv14/kernel")
    pre_gen_dict["stage_2.up.out.4.conv.0.weight"] = torch.tensor(np.transpose(w_allconv14_kernel, [3, 2, 1, 0])).double()
    w_allconv14_bias = reader.get_tensor("inpaint_net/allconv14/bias")
    pre_gen_dict["stage_2.up.out.4.conv.0.bias"] = torch.tensor(w_allconv14_bias) 

    w_allconv15_kernel = reader.get_tensor("inpaint_net/allconv15_upsample/allconv15_upsample_conv/kernel")
    pre_gen_dict["stage_2.up.out.6.conv.0.weight"] = torch.tensor(np.transpose(w_allconv15_kernel, [3, 2, 1, 0])).double()
    w_allconv15_bias = reader.get_tensor("inpaint_net/allconv15_upsample/allconv15_upsample_conv/bias")
    pre_gen_dict["stage_2.up.out.6.conv.0.bias"] = torch.tensor(w_allconv15_bias) 

    w_allconv16_kernel = reader.get_tensor("inpaint_net/allconv16/kernel")
    pre_gen_dict["stage_2.up.out.7.conv.0.weight"] = torch.tensor(np.transpose(w_allconv16_kernel, [3, 2, 1, 0])).double()
    w_allconv16_bias = reader.get_tensor("inpaint_net/allconv16/bias")
    pre_gen_dict["stage_2.up.out.7.conv.0.bias"] = torch.tensor(w_allconv16_bias) 

    w_allconv17_kernel = reader.get_tensor("inpaint_net/allconv17/kernel")
    pre_gen_dict["stage_2.up.out.8.conv.0.weight"] = torch.tensor(np.transpose(w_allconv17_kernel, [3, 2, 1, 0])).double()
    w_allconv17_bias = reader.get_tensor("inpaint_net/allconv17/bias")
    pre_gen_dict["stage_2.up.out.8.conv.0.bias"] = torch.tensor(w_allconv17_bias) 

    print("finish stage_2 upsample")

    # net_state = model.state_dict()                                             
    torch.save(pre_gen_dict, 'G_1_L1_2_pretrained.pth')    


def main():
    tf_model_path = "./generative_inpainting-master/model_logs/release_places2_256/snap-0"
    pytorch_model_path= "./Generative-Inpainting-pytorch-master/models/G_1_L1_2.pth"
    convertWeights(tf_model_path, pytorch_model_path)

main()






# print all weights
# print("=============tensorflow version==================")

# for key in sorted(var_to_shape_map):
#     print("key: ", key)
#     print("weight:", reader.get_tensor(key).shape)


# print("=============pytorch version==================")

# print("++++++generator++++++")
# for k, v in pre_gen_dict.items():
#     print("key: ", k)
#     print("weight:", v.shape)
# print("++++++discriminator++++++")
# for k, v in pre_dis_dict.items():
#     print("key: ", k)
#     print("weight:", v.shape)





# # test accuracy
# reader = pywrap_tensorflow.NewCheckpointReader("./generative_inpainting-master/model_logs/release_places2_256/snap-0")
# var_to_shape_map = reader.get_variable_to_shape_map()
# pre_gen_dict = torch.load( "./Generative-Inpainting-pytorch-master/models/G_1_L1_2.pth")

# sess = tf.Session()

# np.random.seed(1)
# tf.set_random_seed(1)

# kernel_size = 5
# input_feat = 5
# output_feat = 32

# #inputs
# npo = np.random.random((1,5,5, input_feat))
# x_tf = tf.convert_to_tensor(npo, tf.float32) # (1, 5, 5, 5)
# x_torch = torch.tensor(np.transpose(npo, [0, 3, 2, 1])) # torch.Size([1, 5, 5, 5])

# # weights
# # weights = np.random.random((kernel_size,kernel_size,input_feat,output_feat)) 
# # w_tf = tf.Variable(weights, name="testconv_W", dtype=tf.float32)
# # weights_torch = np.transpose(weights, [3, 2, 1, 0]) 

# w_tf = reader.get_tensor("inpaint_net/conv1/kernel")
# w_torch = np.transpose(w_tf, [3, 2, 1, 0]) # (32, 5, 5, 5)

# #convolving with tensorflow
# res_tf = tf.nn.conv2d(x_tf, w_tf, strides=[1, 1, 1, 1], padding="VALID") # (1, 1, 1, 32)

# #convolving with torch
# sess.run(tf.global_variables_initializer())
# res_torch = F.conv2d(x_torch, torch.tensor(w_torch).double(), padding=0, bias=torch.zeros((output_feat)).double())   # torch.Size([1, 32, 1, 1])

# #comparing the results
# print(np.mean(np.transpose(sess.run(res_tf), [0, 3, 1, 2])) - torch.mean(res_torch).detach().numpy())




# tf get weights method 2
# sess = tf.Session()
# saver = tf.train.import_meta_graph('./release_places2_256/snap-0.meta')
# graph = tf.get_default_graph()
# input_graph_def = graph.as_graph_def()
# with tf.Session() as sess:
#     saver.restore(sess, "./release_places2_256/snap-0")
#     print("## Trainable variables: ")
#     for v in tf.trainable_variables():
#         print("variable", v.name)
#         print("value", sess.run(v).shape)

