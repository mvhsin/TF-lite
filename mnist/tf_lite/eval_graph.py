import tensorflow as tf
from src.models import inception_resnet_v1
from src.models import squeezenet
from src.models import yolov3_tiny
import sys
import click
from pathlib import Path


@click.command()
@click.argument('training_checkpoint_dir', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.argument('eval_checkpoint_dir', type=click.Path(exists=True, file_okay=False, resolve_path=True))
def main(training_checkpoint_dir, eval_checkpoint_dir):

    traning_checkpoint = Path(training_checkpoint_dir) / "model-20180204-160909.ckpt-265000"
    eval_checkpoint = Path(eval_checkpoint_dir) / "squeeze_facenet.ckpt"

    data_input = tf.placeholder(name='input', dtype=tf.float32, shape=[None, 28, 28, 1])

    output, _ = squeezenet.inference(data_input, keep_probability=0.8, phase_train=False, bottleneck_layer_size=128)
    output = tf.identity(output, name='output')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        saver = tf.train.Saver()
        saver.restore(sess, traning_checkpoint.as_posix())
        save_path = saver.save(sess, eval_checkpoint.as_posix())
        print("Model saved in file: %s" % save_path)




if __name__ == "__main__":
    main()
