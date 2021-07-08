import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# setting seeds for result reproducibility. This is not super important
tf.set_random_seed(2212)

class DQNAgent:
    def __init__(self, sess, action_dim, observation_dim):
        # Force keras to use the session that we have created
        K.set_session(sess)
        self.sess = sess
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.model = self.create_model()

    def create_model(self):
        state_input = Input(shape=(self.observation_dim))
        # Config1
        # state_h1 = Dense(400, activation='relu')(state_input)
        # state_h2 = Dense(300, activation='relu')(state_h1)
        state_h1 = Dense(64, activation='relu')(state_input)
        state_h2 = Dense(128, activation='relu')(state_h1)
        state_h3 = Dense(256, activation='relu')(state_h2)
        state_h4 = Dense(512, activation='relu')(state_h3)
        state_h10 = Dense(1024, activation='relu')(state_h4)
        state_h5 = Dense(256, activation='relu')(state_h10)
        state_h6 = Dense(128, activation='relu')(state_h5)
        state_h7 = Dense(64, activation='relu')(state_h6)
        state_h8 = Dense(32, activation='relu')(state_h7)
        output = Dense(self.action_dim, activation='linear')(state_h8)
        model = Model(inputs=state_input, outputs=output)
        model.compile(loss='mse', optimizer=Adam(0.005))
        model.summary()
        return model


