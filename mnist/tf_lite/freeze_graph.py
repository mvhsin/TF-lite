import tensorflow as tf
import click
from pathlib import Path

MODEL_DIR = (Path(__file__).parent / 'model').absolute()
OUTPUT_DIR = (Path(__file__).parent / "frozen_graphs").absolute()

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
#logits = conv_net(X, weights, biases, keep_prob)
#prediction = tf.nn.softmax(logits)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    #fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    logits = tf.add(tf.matmul(fc1, weights['out']), biases['out'], name='logits')
    prediction = tf.nn.softmax(logits, name='output')
    return prediction

def freeze_graph_def(sess, input_graph_def, output_node_names):
    # Replace all the variables in the graph with constants of the same values
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","))
    return output_graph_def
    
def save_conv_net(width, height):
        model_path = MODEL_DIR / "model.ckpt"
        graph = tf.Graph()
    #with graph.as_default():
        #define tensor and op in graph(-1,1)
        X = tf.placeholder(tf.float32, [1, num_input], name='input')
        #image_op = tf.placeholder(tf.float32, shape=(1, 28, 28, 1), name='input')
        out = conv_net(X, weights, biases, keep_prob)
        
        #allow 
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, model_path.as_posix())

        # Retrieve the protobuf graph definition and fix the batch norm nodes
        input_graph_def = sess.graph.as_graph_def()
            
        # Freeze the graph def
        output_graph_def = freeze_graph_def(sess, input_graph_def, 'output')
    
        output_conv_net = OUTPUT_DIR / 'conv_net.pb'
        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_conv_net.as_posix(), 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_conv_net))
        
@click.command()
@click.option('--width', default=28)
@click.option('--height', default=28)
def main(width, height):
    save_conv_net(width, height)

if __name__ == "__main__":
    main()
