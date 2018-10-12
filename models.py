

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten


def build_sequential_model():
    """
    11 layers
    - 8 Conv
    - 3 Dense
    
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=10, input_shape=(2000, 4), activation="relu", name="Conv1"))
    model.add(Conv1D(filters=64, kernel_size=10, activation="relu", name="Conv2"))
    model.add(MaxPooling1D(pool_size=5))
    # model.add(Dropout(0.5))

    model.add(Conv1D(filters=128, kernel_size=10, activation="relu", name="Conv3"))
    model.add(Conv1D(filters=128, kernel_size=10, activation="relu", name="Conv4"))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=256, kernel_size=10, activation="relu", name="Conv5"))
    model.add(Conv1D(filters=256, kernel_size=10, activation="relu", name="Conv6"))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=256, kernel_size=10, activation="relu", name="Conv7"))
    model.add(Conv1D(filters=256, kernel_size=10, activation="relu", name="Conv8"))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.5))

    model.add(Flatten(name="Flatten1"))

    model.add(Dense(625, activation='relu', name='Dense1'))
    model.add(Dense(625, activation='relu', name='Dense2'))
    model.add(Dense(125, activation='relu', name='Dense3'))
    model.add(Dense(1, activation="linear", name='Dense4'))

    return model


from keras import Model
from keras.utils import multi_gpu_model


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model
    
    def __getattribute__(self, attrname):
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)
        
        return super(ModelMGPU, self).__getattribute__(attrname)
