import tensorflow.keras.utils as utils
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers.legacy as optimizers


train = utils.image_dataset_from_directory(
    'images',
    label_mode = 'categorical',
    batch_size = 32,                   # Default
    image_size = (600, 600),
    seed = 29,
    validation_split = 0.3,
    subset = 'training',
)

test = utils.image_dataset_from_directory(
    'images',
    label_mode = 'categorical',
    batch_size = 32,                   # Default
    image_size = (600, 600),
    seed = 29,
    validation_split = 0.3,
    subset = 'validation',
)

class Net():
    def __init__(self, input_shape):
        self.model = models.Sequential()
        # self.model.add(layers.ZeroPadding2D(
        #   padding = (1,0), (1,0), # top bottom left right
        #    input_shape = input_shape, # move this from bottom
        # )
        self.model.add(layers.Conv2D(
            8, # filters
            30, # kernel
            strides = 10, # step size
            activation = 'relu',
            input_shape = input_shape,  # only need this on first layer
        ))  # Output 58 x 58 x 8
        self.model.add(layers.MaxPool2D(
            pool_size = 2,  # strides default is pool_size
        )) # Output: 29 x 29 x 8
        self.model.add(layers.ZeroPadding2D(
            padding = ((1,0), (1,0)), # top bottom left right
            input_shape = input_shape, # move this from bottom
        )) 
        self.model.add(layers.Conv2D(
            8, # filters
            3, # kernel
            strides = 1,
            activation = 'relu'
        )) # Output: 28 x 28 x 8
        self.model.add(layers.MaxPool2D(
            pool_size = 2,
        )) # Output: 14 x 14 x 8 
        self.model.add(layers.Flatten(    
        )) # Output: 1568
        self.model.add(layers.Dense(
            512,
            activation = 'relu',
        ))
        self.model.add(layers.Dense(
            128,
            activation = 'relu',
        ))
        self.model.add(layers.Dense(
            32,
            activation = 'relu',
        ))
        self.model.add(layers.Dense(
            16, # Exactly equal to number of classes
            activation = 'softmax', # Always use softmax on last layer
        ))
        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy'],
        )

        def __str__(self):
            self.model.summary()
            return ""

net = Net((600, 600, 3))

net.model.fit(
    train,
    batch_size = 32,    # Bigger than class size so you more likely grab all classes
    epochs = 10,
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
)

net.model.save('faces_model_save')







