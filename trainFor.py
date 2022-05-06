import os
import socket
import threading
import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import io
import imageio

# We'll define a helper function to shift the frames, where`x` is frames 0 to n - 1, and `y` is frames 1 to n.
def create_shifted_frames(data,descrip):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    # Inspect the dataset.
    print(descrip+ str(x.shape) + ", " + str(y.shape))
    return x, y

def makeDataSet():
    # Download and load the dataset.
    fpath = keras.utils.get_file("moving_mnist.npy",
                                 "file:///../trainModel/mnist_test_seq.npy", )
    dataset = np.load(fpath)
    # Swap the axes representing the number of frames and number of data samples.
    dataset = np.swapaxes(dataset, 0, 1)
    # We'll pick out 1000 of the 10000 total examples and use those.
    dataset = dataset[:1000, ...]
    # Add a channel dimension since the images are grayscale.
    dataset = np.expand_dims(dataset, axis=-1)
    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)

    train_index = indexes[: int(0.9 * dataset.shape[0])]
    val_index = indexes[int(0.9 * dataset.shape[0]):]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    # Normalize the data to the 0-1 range.
    train_dataset = train_dataset / 255
    val_dataset = val_dataset / 255

    return train_dataset,val_dataset

def generate_movies(n_samples=1200, n_frames=20):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)

    for i in range(n_samples):
        # 添加 5 到 7 个移动方块
        n = np.random.randint(5, 8)

        for j in range(n):
            # 初始位置
            xstart = np.random.randint(10, 74)
            ystart = np.random.randint(10, 74)
            # 运动方向
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # 方块尺寸
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w : x_shift + w, y_shift - w : y_shift + w, 0] += 1

                # 通过添加噪音使其更加健壮。
                # 这个想法是，如果在推理期间，像素的值不是一个，
                # 我们需要训练更加健壮的网络，并仍然将其视为属于方块的像素。
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    noisy_movies[i, t, x_shift - w - 1: x_shift + w + 1, y_shift - w - 1: y_shift + w + 1, 0] += noise_f * 0.1

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w, 0] += 1

    # 裁剪为 40x40 窗口
    noisy_movies = noisy_movies[::, ::, 10:74, 10:74, ::]
    shifted_movies = shifted_movies[::, ::, 10:74, 10:74, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies

class modeltrainThread (threading.Thread):
    def __init__(self, threadID, name, delay):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.delay = delay
    def run(self):
        print ("开始线程：" + self.name)
        modeltrainfunc(self.name, self.delay, 5)
        print ("退出线程：" + self.name)

def modeltrainfunc(threadName, delay, counter):
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 9999
    serversocket.bind((HOST, port))
    serversocket.listen(5)
    while True:
        clientsocket, addr = serversocket.accept()
        print("连接地址: %s" % str(addr))
        data = clientsocket.recv(1024).decode('utf-8')
        # TODO:获取参数
        modelArgs = json.loads(data)
        aid = modelArgs['aid']
        k1 = modelArgs['k1']
        k2 = modelArgs['k2']
        k3 = modelArgs['k3']
        filters = modelArgs['filters']
        optimizer = modelArgs['optimizer']
        batchSize = modelArgs['batchSize']
        epochs = modelArgs['epochs']
        loss = modelArgs['loss']
        state = modelArgs['state']
        model = modeltrain(filters,k1,k2,k3,batchSize,epochs)
        try:
            myURL = urllib.request.urlopen("http://localHOST:8080/arredy?aid=" + str(aid))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(404)

        clientsocket.close()

def modeltrain(f, k1, k2, k3, bs, ep):
    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=(None, *x_train.shape[2:]))
    # We will construct 3 `ConvLSTM2D` layers with batch normalization,followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(filters=f, kernel_size=(k1, k1),
                          padding="same", return_sequences=True, activation="relu", )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=f, kernel_size=(k2, k2),
                          padding="same", return_sequences=True, activation="relu", )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=f, kernel_size=(k3, k3),
                          padding="same", return_sequences=True, activation="relu", )(x)
    x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3),
                      activation="sigmoid", padding="same")(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())

    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
    modelcheckpoint_mcp = keras.callbacks.ModelCheckpoint(filepath=model_save_dir, save_freq='epoch')
    tensorboard_tb = keras.callbacks.TensorBoard(log_dir='./logs')

    # Define modifiable training hyperparameters.
    # Fit the model to the training data.
    model.fit(x_train, y_train, batch_size=bs, epochs=ep, validation_data=(x_val, y_val),
              callbacks=[early_stopping, reduce_lr, modelcheckpoint_mcp, tensorboard_tb], )
    return model

class imgcreateThread (threading.Thread):
    def __init__(self, threadID, name, delay):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.delay = delay
    def run(self):
        print ("开始线程：" + self.name)
        imgcreatefunc(self.name, self.delay, 5)
        print ("退出线程：" + self.name)

def savetheOriginTruth(data_choice,urlList):
    # Construct a figure on which we will visualize the images.
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(np.squeeze(val_dataset[data_choice[0]][idx]), cmap="gray")
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")
    # Print information and display the figure.
    print(f"Displaying frames for example {data_choice[0]}.")
    plt.savefig(save_dir+"/theOriginTruth.png")
    urlList.append("theOriginTruth.png")
    plt.show()

def savethePredict(data_choice,urlList,model):
    # Select a random example from the validation dataset.
    example = val_dataset[data_choice[0]]
    # Pick the first/last ten frames from the example.
    frames = example[:10, ...]
    original_frames = example[10:, ...]
    # Predict a new set of 10 frames.
    for _ in range(10):
        # Extract the model's prediction and post-process it.
        new_prediction = model.predict(np.expand_dims(frames, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
        # Extend the set of prediction frames.
        frames = np.concatenate((frames, predicted_frame), axis=0)
    # Construct a figure for the original and new frames.
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    # Plot the original frames.
    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")
    # Plot the new frames.
    new_frames = frames[10:, ...]
    for idx, ax in enumerate(axes[1]):
        ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")
    # Display the figure.
    plt.savefig(save_dir+"/thePredict.png")
    urlList.append("thePredict.png")
    plt.show()

def savetheVedios(data_choice,urlList,model):
    # Select a few random examples from the dataset.
    examples = val_dataset[data_choice]
    eidx = 0
    # Iterate over the examples and predict the frames.
    predicted_videos = []
    for example in examples:
        # Pick the first/last ten frames from the example.
        frames = example[:10, ...]
        original_frames = example[10:, ...]
        new_predictions = np.zeros(shape=(10, *frames[0].shape))
        # Predict a new set of 10 frames.
        for i in range(10):
            # Extract the model's prediction and post-process it.
            frames = example[: 10 + i + 1, ...]
            new_prediction = model.predict(np.expand_dims(frames, axis=0))
            new_prediction = np.squeeze(new_prediction, axis=0)
            predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
            # Extend the set of prediction frames.
            new_predictions[i] = predicted_frame
        # Create and save GIFs for each of the ground truth/prediction images.
        for frame_set in [original_frames, new_predictions]:
            eidx += 1
            # Construct a GIF from the selected video frames.
            current_frames = np.squeeze(frame_set)
            current_frames = current_frames[..., np.newaxis] * np.ones(3)
            current_frames = (current_frames * 255).astype(np.uint8)
            current_frames = list(current_frames)
            # Construct a GIF from the frames.
            with io.BytesIO() as gif:
                urlList.append(f'{eidx}_check.gif')
                imageio.mimsave(save_dir+f'/{eidx}_check.gif', current_frames, "GIF", fps=5)
                imageio.mimsave(gif, current_frames, "GIF", fps=5)
                predicted_videos.append(gif.getvalue())

def savetheMovies(urlList,model):
    noisy_movies, shifted_movies = generate_movies(n_samples=300)
    which = 200
    track = noisy_movies[which][:10, ::, ::, ::]
    for j in range(20):
        new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::])
        new = new_pos[::, -1, ::, ::, ::]
        track = np.concatenate((track, new), axis=0)
    # 然后将预测与实际进行比较
    track2 = noisy_movies[which][::, ::, ::, ::]
    for i in range(20):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        if i >= 10:
            ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
        else:
            ax.text(1, 3, 'Initial trajectory', fontsize=20)
        toplot = track[i, ::, ::, 0]
        plt.imshow(toplot)
        ax = fig.add_subplot(122)
        plt.text(1, 3, 'Ground truth', fontsize=20)
        toplot = track2[i, ::, ::, 0]
        if i >= 2:
            toplot = shifted_movies[which][i - 1, ::, ::, 0]
        plt.imshow(toplot)
        plt.savefig(save_dir+f"/{i + 1}_animate.png")
        plt.show()
    image_list = [r''+save_dir+f'/{str(x)}_animate.png' for x in range(1, 20)]
    gif_name = r''+save_dir+'/animate.gif'
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread_v2(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.3)
    urlList.append('animate.gif')

def imgcreatefunc(threadName, delay, counter):
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 8888
    serversocket.bind((HOST, port))
    serversocket.listen(5)
    while True:
        clientsocket, addr = serversocket.accept()
        print("连接地址: %s" % str(addr))
        print(clientsocket.recv(1024).decode('UTF-8'))
        model = keras.models.load_model(model_save_dir)

        # Plot each of the sequential images for one random data example.
        data_choice = np.random.choice(range(len(val_dataset)), size=5)
        urlList = []
        savetheOriginTruth(data_choice, urlList)
        savethePredict(data_choice, urlList, model)
        savetheVedios(data_choice, urlList, model)
        savetheMovies(urlList, model)

        imgUrlJson = json.dumps(urlList)
        clientsocket.send(imgUrlJson.encode('utf-8'))
        clientsocket.close()

def makeSaveDir(pathto):
    if not os.path.isdir(pathto):
        os.makedirs(pathto)
    return pathto

HOST = "127.0.0.1"

save_dir = makeSaveDir(pathto='../UsingConvLSTM/src/main/resources/static/img')
model_save_dir = makeSaveDir(pathto='./pathtosavemodel/model')

train_dataset,val_dataset = makeDataSet()
# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_dataset,descrip="Training Dataset Shapes: ")
x_val, y_val = create_shifted_frames(val_dataset,descrip="Validation Dataset Shapes: ")

model = modeltrain(f=40,k1=5,k2=3,k3=1,bs=1,ep=1)