from tensorflow.keras import layers
from tensorflow.keras import Input
import tensorflow as tf
import numpy as np

def make_model(c, rb1=True, rb2=False, rb3=False):
    inputs = Input(shape=(1,c))
    
    
    if rb1:
        # Residual block 1
        y = layers.Conv1D(filters=2, kernel_size=17, strides=1, padding='same')(inputs)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        
        y = layers.Conv1D(filters=4, kernel_size=11, strides=1, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        
        y = layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='same')(y)
        block_1_output = layers.BatchNormalization()(y)
        
        #skipped layer
        z = layers.Conv1D(filters=8, kernel_size=1, strides=1, padding='same')(inputs)
        z = layers.BatchNormalization()(z)
        
        block_2_output = layers.add([z, block_1_output])    
        x = layers.ReLU()(block_2_output)
        residual_block_1_output = layers.AveragePooling1D(pool_size=5, padding='same')(x)
    
    else:
        residual_block_1_output = inputs    
    
    
    if rb2:
        # Residual block 2
        
        y = layers.Conv1D(filters=2, kernel_size=17, strides=1, padding='same')(residual_block_1_output)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        
        y = layers.Conv1D(filters=4, kernel_size=11, strides=1, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        
        y = layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='same')(y)
        block_2_output = layers.BatchNormalization()(y)
        
        #skipped layer
        z = layers.Conv1D(filters=8, kernel_size=1, strides=1, padding='same')(residual_block_1_output)
        z = layers.BatchNormalization()(z)
        
        block_3_output = layers.add([z, block_2_output])    
        x = layers.ReLU()(block_3_output)
        residual_block_2_output = layers.AveragePooling1D(pool_size=5, padding='same')(x)
    
    else:
        residual_block_2_output = residual_block_1_output
    
    
    if rb3:
        # Residual block 3
        
        y = layers.Conv1D(filters=2, kernel_size=17, strides=1, padding='same')(residual_block_2_output)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        
        y = layers.Conv1D(filters=4, kernel_size=11, strides=1, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        
        y = layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='same')(y)
        block_3_output = layers.BatchNormalization()(y)
        
        #skipped layer
        z = layers.Conv1D(filters=8, kernel_size=1, strides=1, padding='same')(residual_block_2_output)
        z = layers.BatchNormalization()(z)
        
        block_4_output = layers.add([z, block_3_output])    
        x = layers.ReLU()(block_4_output)
        residual_block_3_output = layers.AveragePooling1D(pool_size=5, padding='same')(x)
    
    else:
        residual_block_3_output = residual_block_2_output

    
    
    x = layers.GlobalAveragePooling1D()(residual_block_3_output)
    outputs = layers.Dropout(0.5)(x)
    
    return tf.keras.Model(inputs, outputs)
    
def dnn_model(data, rb1=True, rb2=False, rb3=False):
    '''
    data: input training data
    rb1: enable / disable residual block 1
    rb2: enable / disable residual block 2
    rb3: enable / disable residual block 3        
    '''

    a,b,c = np.shape(data)
    
    in1 = Input(shape=(1,c), name="ecg1")
    in2 = Input(shape=(1,c), name="ecg2")
    in3 = Input(shape=(1,c), name="ecg3")
    in4 = Input(shape=(1,c), name="ecg4")
    in5 = Input(shape=(1,c), name="ecg5")
    in6 = Input(shape=(1,c), name="ecg6")
    in7 = Input(shape=(1,c), name="ecg7")
    in8 = Input(shape=(1,c), name="ecg8")
    in9 = Input(shape=(1,c), name="ecg9")
    in10 = Input(shape=(1,c), name="ecg10")
    in11 = Input(shape=(1,c), name="ecg11")
    in12 = Input(shape=(1,c), name="ecg12")


    model1 = make_model(c, rb1, rb2, rb3)(in1)
    model2 = make_model(c, rb1, rb2, rb3)(in2)
    model3 = make_model(c, rb1, rb2, rb3)(in3)
    model4 = make_model(c, rb1, rb2, rb3)(in4)
    model5 = make_model(c, rb1, rb2, rb3)(in5)
    model6 = make_model(c, rb1, rb2, rb3)(in6)
    model7 = make_model(c, rb1, rb2, rb3)(in7)
    model8 = make_model(c, rb1, rb2, rb3)(in8)
    model9 = make_model(c, rb1, rb2, rb3)(in9)
    model10 = make_model(c, rb1, rb2, rb3)(in10)
    model11 = make_model(c, rb1, rb2, rb3)(in11)
    model12 = make_model(c, rb1, rb2, rb3)(in12)

    fused_layer = layers.concatenate([model1, model2, model3, model4, model5, model6, model7,
                                      model8, model9, model10, model11, model12])

    fused_layer = layers.Flatten()(fused_layer)
    fused_layer = layers.Dense(32, activation='relu')(fused_layer)
    outputs = layers.Dense(1, activation='softmax', name="label")(fused_layer)

    return tf.keras.Model(inputs=[in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12],
                              outputs=[outputs])

default_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=10)

def call(train_X, train_y, test_X, test_y, optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"], 
    epochs=60, batch_size=128, callbacks=default_callback, rb1=True, rb2=False, rb3=False):
    '''
    train_X: input training data
    train_y: input training label
    test_X:  input testing data   
    test_y:  input testing label   
    optimizer='adam': optimizer 
    loss='binary_crossentropy': loss function 
    metrics=["accuracy"]: metrics to monitor 
    epochs=60, 
    batch_size=128, 
    callbacks=default_callback, 
    rb1: enable / disable residual block 1
    rb2: enable / disable residual block 2
    rb3: enable / disable residual block 3 
    '''

    a,b,c = np.shape(train_X)
    
    final_model = dnn_model(data=train_X, 
                        rb1=True, rb2=False, rb3=False)

    tf.keras.utils.plot_model(final_model, 'dnn.jpg', show_shapes=True)
    
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, name='SGD')
    # loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    final_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    history = final_model.fit(
        {"ecg1": train_X[:,0,:].reshape(-1,1,c),"ecg2": train_X[:,1,:].reshape(-1,1,c),"ecg3": train_X[:,2,:].reshape(-1,1,c),
         "ecg4": train_X[:,3,:].reshape(-1,1,c),"ecg5": train_X[:,4,:].reshape(-1,1,c),"ecg6": train_X[:,5,:].reshape(-1,1,c),
         "ecg7": train_X[:,6,:].reshape(-1,1,c),"ecg8": train_X[:,7,:].reshape(-1,1,c),"ecg9": train_X[:,8,:].reshape(-1,1,c),
         "ecg10": train_X[:,9,:].reshape(-1,1,c),"ecg11": train_X[:,10,:].reshape(-1,1,c),"ecg12": train_X[:,11,:].reshape(-1,1,c)},
        {"label": train_y},
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
        )
    print("\n\nEvaluate on test data:")    
    results = final_model.evaluate(
        {"ecg1": test_X[:,0,:].reshape(-1,1,c),"ecg2": test_X[:,1,:].reshape(-1,1,c),"ecg3": test_X[:,2,:].reshape(-1,1,c),
         "ecg4": test_X[:,3,:].reshape(-1,1,c),"ecg5": test_X[:,4,:].reshape(-1,1,c),"ecg6": test_X[:,5,:].reshape(-1,1,c),
         "ecg7": test_X[:,6,:].reshape(-1,1,c),"ecg8": test_X[:,7,:].reshape(-1,1,c),"ecg9": test_X[:,8,:].reshape(-1,1,c),
         "ecg10": test_X[:,9,:].reshape(-1,1,c),"ecg11": test_X[:,10,:].reshape(-1,1,c),"ecg12": test_X[:,11,:].reshape(-1,1,c)},
        {"label": test_y})
    print("Test loss, Test acc:", results)



