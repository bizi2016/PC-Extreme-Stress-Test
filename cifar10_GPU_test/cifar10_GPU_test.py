import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


from tensorflow import keras

# 归一化数据
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# 独热编码
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 构造模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout


def get_model():
    
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


model = get_model()
    
# 编译并打印模型
model.compile( optimizer=keras.optimizers.Adam(learning_rate=1e-4),
               loss='categorical_crossentropy',
               metrics=['accuracy'],
               )

model.summary()


# 打印准确率和loss
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 300  # 图片像素

class PlotProgress(keras.callbacks.Callback):

    def __init__(self, entity = ['loss', 'accuracy']):
        self.entity = entity

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        
        self.losses = []
        self.val_losses = []

        self.accs = []
        self.val_accs = []

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        # 损失函数
        self.losses.append(logs.get('{}'.format(self.entity[0])))
        self.val_losses.append(logs.get('val_{}'.format(self.entity[0])))
        # 准确率
        self.accs.append(logs.get('{}'.format(self.entity[1])))
        self.val_accs.append(logs.get('val_{}'.format(self.entity[1])))

        self.i += 1
        
        plt.figure( figsize = (6, 3) )

        plt.subplot(121)
        plt.plot(self.x, self.losses, label="{}".format(self.entity[0]))
        plt.plot(self.x, self.val_losses, label="val_{}".format(self.entity[0]))
        plt.legend()
        plt.title('loss')

        plt.subplot(122)
        plt.plot(self.x, self.accs, label="{}".format(self.entity[1]))
        plt.plot(self.x, self.val_accs, label="val_{}".format(self.entity[1]))
        plt.legend()
        plt.title('accuracy')

        plt.tight_layout()  # 减少白边
        plt.savefig('vis.jpg')
        plt.close()  # 关闭


# 绘图函数
plot_progress = PlotProgress(entity=['loss', 'accuracy'])

# 早产
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)


model.fit( train_images, train_labels,
           validation_data=(test_images, test_labels),

           epochs=1000, batch_size=32,

           callbacks=[plot_progress, early_stopping],

           verbose=1,  # 2 一次训练就显示一行

           shuffle=True,  # 再次打乱
           )
