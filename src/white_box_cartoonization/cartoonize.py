import cv2
import numpy as np
import tensorflow as tf

import white_box_cartoonization.guided_filter as guided_filter
import white_box_cartoonization.network as network


class Cartoonizer():
    def __init__(self, model_path="./data/cartoonize_models"):  # TODO support multiple
        self.input_photo = tf.placeholder(tf.float32, [None, None, None, 3])
        network_out = network.unet_generator(self.input_photo)
        self.final_out = guided_filter.guided_filter(self.input_photo, network_out, r=1, eps=5e-3)

        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if 'generator' in var.name]
        saver = tf.train.Saver(var_list=gene_vars)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, model_path.replace(".index", ""))

    @staticmethod
    def resize_crop(image):
        h, w, c = np.shape(image)
        if min(h, w) > 720:
            if h > w:
                h, w = int(720 * h / w), 720
            else:
                h, w = 720, int(720 * w / h)
        image = cv2.resize(image, (w, h),
                           interpolation=cv2.INTER_AREA)
        h, w = (h // 8) * 8, (w // 8) * 8
        image = image[:h, :w, :]
        return image

    def stylize(self, frame):
        frame_list = [frame]  # Kept it with a list to make it easily adaptable for larger bach sizes
        frame_list = [np.expand_dims(self.resize_crop(f), axis=0) for f in frame_list]
        frame_tensor = np.concatenate(frame_list, axis=0)

        batch_image = frame_tensor.astype(np.float32) / 127.5 - 1
        output = self.sess.run(self.final_out, feed_dict={self.input_photo: batch_image})
        output = (output + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = [output[r, :, :, :] for r in range(output.shape[0])][0]
        return output

# def cartoonize(load_folder, save_folder, model_path):
#     input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
#     network_out = network.unet_generator(input_photo)
#     final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)
#
#     all_vars = tf.trainable_variables()
#     gene_vars = [var for var in all_vars if 'generator' in var.name]
#     saver = tf.train.Saver(var_list=gene_vars)
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)
#
#     sess.run(tf.global_variables_initializer())
#     saver.restore(sess, tf.train.latest_checkpoint(model_path))
#     name_list = os.listdir(load_folder)
#     for name in tqdm(name_list):
#         try:
#             load_path = os.path.join(load_folder, name)
#             save_path = os.path.join(save_folder, name)
#             image = cv2.imread(load_path)
#             image = _resize_crop(image)
#             batch_image = image.astype(np.float32)/127.5 - 1
#             batch_image = np.expand_dims(batch_image, axis=0)
#             output = sess.run(final_out, feed_dict={input_photo: batch_image})
#             output = (np.squeeze(output)+1)*127.5
#             output = np.clip(output, 0, 255).astype(np.uint8)
#             cv2.imwrite(save_path, output)
#         except:
#             print('cartoonize {} failed'.format(load_path))
#
#
#
#
# if __name__ == '__main__':
#     model_path = 'saved_models'
#     load_folder = 'test_images'
#     save_folder = 'cartoonized_images'
#     if not os.path.exists(save_folder):
#         os.mkdir(save_folder)
#     cartoonize(load_folder, save_folder, model_path)
#
