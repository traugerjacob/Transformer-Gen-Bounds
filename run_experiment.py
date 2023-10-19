import tensorflow as tf
import numpy as np
import sys

class SparseMajorityDataset():
    def __init__(self, dimension=5, T=6, s=3, samples=10000, pe_size=None, s_idxs=None, unique=False):
        assert T>=s
        assert s%2 == 1
        if s_idxs is not None:
            assert len(s_idxs) == s
        self.embed_dim = dimension
        self.seq_length = T
        self.s = s
        self.num_samples = samples
        self.sparse_pos = s_idxs
        self.unique = unique
        if pe_size is None:
            self.pe_size = self.seq_length
        else:
            self.pe_size = pe_size
        self.make_data()

    def make_ortho_embeddings(self):
        self.zero_embed = np.random.uniform(low=-1,high=1,size=self.embed_dim)
        self.one_embed = np.random.uniform(low=-1,high=1,size=self.embed_dim)
        self.one_embed -= np.dot(self.one_embed, self.zero_embed)*self.zero_embed/np.dot(self.zero_embed,self.zero_embed)

        self.cls = np.random.uniform(low=-1,high=1,size=self.embed_dim)
        self.cls -= np.dot(self.cls, self.zero_embed)*self.zero_embed/np.dot(self.zero_embed,self.zero_embed)
        self.cls -= np.dot(self.cls, self.one_embed)*self.one_embed/np.dot(self.one_embed,self.one_embed)

        self.one_embed = self.one_embed/np.linalg.norm(self.one_embed, ord=1)
        self.zero_embed = self.zero_embed/np.linalg.norm(self.zero_embed, ord=1)
        self.cls = self.cls/np.linalg.norm(self.cls, ord=1)

    def make_labels(self):
        self.labels = 1*(np.sum(self.non_embedded_data[:, self.sparse_pos], axis=1) >=self.s/2)
        self.one_hot_labels = tf.one_hot(self.labels,2)
        
    def make_data(self):
        if self.sparse_pos is None:
            self.sparse_pos = np.random.choice(self.seq_length, self.s, replace=False)
        self.non_embedded_data = np.random.choice([0,1], (self.num_samples, self.seq_length))
        if self.unique:
            self.make_non_embed_data_unique()
        self.make_ortho_embeddings()
        self.make_labels()
        self.make_pe()
        self.data = np.empty((self.num_samples, self.seq_length+1, self.embed_dim))
        for i,sample in enumerate(self.non_embedded_data):
            data_point = np.empty((self.seq_length+1,self.embed_dim))
            for j,point in enumerate(sample):
                if point == 0:
                    data_point[j,:] = self.zero_embed
                else:
                    data_point[j,:] = self.one_embed
            data_point[self.seq_length,:] = self.cls
            self.data[i,:,:] = data_point + self.pe


    def make_non_embed_data_unique(self):
        sett = set()
        for i in self.non_embedded_data:
            val = ""
            for j in i:
                val += str(j)
            sett.add(val)
        while len(sett) < self.num_samples:
            new_val = "".join(np.random.choice(["0","1"], self.seq_length))
            sett.add(new_val)
        for i,sample in enumerate(list(sett)):
            for j,char in enumerate(sample):
                self.non_embedded_data[i,j] = int(char)



    def make_pe(self):
        self.pe = np.zeros((self.pe_size+1, self.embed_dim))
        for i in range(self.embed_dim):
            sinusoid = np.zeros(self.pe_size+1)
            if i%2 == 0:
                for j in range(self.pe_size+1):
                    sinusoid[j] = np.sin(j/10000**(i/self.embed_dim))
            else:
                for j in range(self.pe_size+1):
                    sinusoid[j] = np.cos(j/10000**(i/self.embed_dim))

            self.pe[:,i] = sinusoid
        self.pe = self.pe[:self.seq_length+1,:]
        return self.pe


class Last_Linear(tf.keras.layers.Layer):
    def __init__(self, output_dim=2, input_dim=32, name="linear"):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, output_dim), initializer="random_normal", trainable=True, name="linear_matrix"
        )
        self.b = self.add_weight(shape=(output_dim,), initializer="zeros", trainable=True, name="linear_bias")

    def call(self, inputs):
        
        return tf.matmul(inputs[:,-1,:], self.w) + self.b

class Transformer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(heads,embed_dim)
        self.relu = tf.keras.layers.ReLU()
        self.dense = tf.keras.layers.Dense(e_d)

    def call(self, inputs):
        mha_out = self.mha(inputs, inputs)
        relu_out = self.relu(mha_out)
        dense_out = self.dense(relu_out)
        return dense_out


class PerfectTrainAccCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=1):
        super().__init__()
        self.patience = patience
        self.add_to_counter = False
        self.counter = 0
        self.total_epochs = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.total_epochs += 1
        if logs["accuracy"] == 1.0:
            self.add_to_counter = True
            self.counter += 1
        else:
            if self.add_to_counter:
                self.add_to_counter = False
                self.counter = 0
        if self.counter >= self.patience:
            self.model.stop_training = True

def scheduler(epoch, lr):
    lr_new = lr*.999#*.99995
    return max(.001, lr_new)



T = int(sys.argv[1])
run = int(sys.argv[3])
e_d = 64
s=9
heads=2
num_train_samples = 300
num_test_samples = 10000
num_samples = num_train_samples + num_test_samples
s_idxs = np.random.choice(T, s, replace=False)
data = SparseMajorityDataset(dimension=e_d, T=T ,s=s, samples=num_samples, s_idxs=s_idxs)
train_data = data.data[:num_train_samples]
train_labels = data.labels[:num_train_samples]
val_data = data.data[num_train_samples:]
val_labels = data.labels[num_train_samples:]

model = tf.keras.Sequential()
model.add(Transformer(e_d,heads))
model.add(Last_Linear(input_dim=e_d))

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=.01, momentum=.9), loss=bce, metrics=['accuracy'])

lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)
checkpoint_filepath = f"./{str(T)}/{str(run)}/chkpt"
mcc = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1)
acc = PerfectTrainAccCallback()

bs = int(sys.argv[2])
model.fit(train_data, tf.one_hot(train_labels,2), epochs=200000, batch_size=bs, validation_data=(val_data, tf.one_hot(val_labels,2)), validation_freq=1, callbacks=[mcc, acc, lrs], verbose=2)
model.load_weights(checkpoint_filepath)
print("best weights loaded")
train_loss, train_acc = model.evaluate(x=train_data, y=tf.one_hot(train_labels,2), verbose=2)
test_loss, test_acc = model.evaluate(x=val_data, y=tf.one_hot(val_labels,2), verbose=2)
model_norms_trained = {}
mn_sum = 0
for j in model.trainable_variables:
    if model_norms_trained.get(j.name) == None:
        model_norms_trained[j.name] = float(tf.norm(j, ord=1))
        mn_sum += float(tf.norm(j, ord=1))
        print(f'{j.name}: {model_norms_trained[j.name]}')
print("model_norms:")
print(model_norms_trained)
print()
print("total weights norm sum:")
print(mn_sum)
print("accuracy at best val_loss (train, test):")
print((train_acc, test_acc))
print("accuracy gen gap (train - test):")
print(train_acc - test_acc)
print("loss at best val_loss (train, test):")
print((train_loss, test_loss))
print("loss gen gap (train - test):")
print(train_loss - test_loss)

