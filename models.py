from tensorflow.keras import layers, models

# selfmade standard CNN without batchnorm but with pooling
def own_custom_conv_net(input_shape):
    model = models.Sequential([
        # Input layer (not explicitly needed in Sequential, but shows input shape)
        layers.InputLayer(shape=input_shape),
        
        # Add some convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten the output
        layers.Flatten(),
        
        # Dense layers
        layers.Dense(64, activation='relu'),

        # Two output layers for both types
        layers.Dense(18, activation='sigmoid'), 
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss= 'binary_crossentropy',
        metrics=['accuracy']
        )
    
    return model

# model from the linked github repo https://github.com/hemagso/Neuralmon from https://jgeekstudies.org/2017/03/12/who-is-that-neural-network/ 
def neuralmon_conv_net(input_shape):
    #Defining model Architecture
    model = models.Sequential([
            layers.Convolution2D(32, 5, 5, input_shape=input_shape),
            layers.Activation("relu"),
            layers.MaxPooling2D((2,2)),
            
            layers.Convolution2D(64, 5, 5),
            layers.Activation("relu"),
            layers.MaxPooling2D((2,2)),
        
            layers.Flatten(),
            
            layers.Dense(64),
            layers.Activation("relu"),
            layers.Dense(18),
            layers.Activation("sigmoid")
        ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model

# model from: https://www.linkedin.com/pulse/gotta-catch-em-all-pokémon-types-ai-recognition-maciej-gajewski/ with sigmoid output layer instead of softmax
def smaller_VGGNet(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape), 
        layers.BatchNormalization(axis=1), 
        layers.MaxPooling2D((3, 3)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'), 
        layers.BatchNormalization(axis=1),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'), 
        layers.BatchNormalization(axis=1),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25), 

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'), 
        layers.BatchNormalization(axis=1),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'), 
        layers.BatchNormalization(axis=1),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), padding='same', activation='relu'), 
        layers.BatchNormalization(axis=1),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'), 
        layers.BatchNormalization(axis=1),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
            
        layers.Dense(1024, activation = 'relu'),
        layers.BatchNormalization(axis=1),
        layers.Dropout(0.5),

        layers.Dense(18, activation='sigmoid'),
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model

# custom CNN from https://github.com/rshnn/pokemon-types/blob/master/02-model-building.ipynb
def custom_CNN_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape), 
        layers.BatchNormalization(), 
        layers.MaxPooling2D((2, 2)),   
        
        layers.Conv2D(32, (3, 3), activation='relu'), 
        layers.BatchNormalization(), 
        layers.MaxPooling2D((2, 2)), 
        
        layers.Conv2D(64, (3, 3), activation='relu'), 
        layers.BatchNormalization(), 
        layers.MaxPooling2D((2, 2)), 

        layers.Conv2D(128, (3, 3), activation='relu'), 
        layers.BatchNormalization(), 
        layers.MaxPooling2D((2, 2)), 
        
        layers.Conv2D(150, (3, 3), activation='relu'), 
        layers.BatchNormalization(), 
        layers.MaxPooling2D((2, 2)), 
        
        layers.Flatten(), 
        layers.Dense(64, activation='relu'), 
        layers.Dense(18, activation='sigmoid'), 
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model