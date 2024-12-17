from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape
train_labels
test_images.shape
len(test_labels)
test_labels