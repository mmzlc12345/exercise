#import modules
import sys
import tensorflow as tf
from PIL import Image, ImageFilter
import numpy as np


checkpoint_dir = "./model/"
model_path = tf.train.latest_checkpoint(checkpoint_dir)
meta_path = model_path+".meta"
saver = tf.train.import_meta_graph(meta_path,True)
init_op = tf.global_variables_initializer()
sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

def predictint(imvalue):
    with tf.Session(config=sess_config) as sess:
        sess.run(init_op)
        saver.restore(sess, model_path)
        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name("input:0")
        predict = graph.get_tensor_by_name("result:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        return sess.run(predict, feed_dict={keep_prob: 1.0, input: [imvalue]})


def imageprepare(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))
    if width > height:
        nheight = int(round((20.0/width*height),0))
        if (nheight == 0):
            nheigth = 1
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0/height*width),0))
        if (nwidth == 0):
            nwidth = 1
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0))
        newImage.paste(img, (wleft, 4))
    tv = list(newImage.getdata())
    tva = [ (255-x)*1.0/255.0 for x in tv]
    return tva


def main(argv):
    imvalue = imageprepare(argv)
    predint = predictint(imvalue)
    print(np.argmax(predint[0]))

if __name__ == "__main__":
    main(sys.argv[1])