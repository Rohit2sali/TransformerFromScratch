from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, function
from tensorflow.keras.losses import sparse_categorical_crossentropy
from transformer import TransformerModel
from prepareDataset import PerpareDateset
from time import time
from pickle import dump


# Define the model parameters
h = 8 # Number of self-attention heads
d_k = 64 # Dimensionality of the linearly projected queries and keys
d_v = 64 # Dimensionality of the linearly projected values
d_model = 512 # Dimensionality of model layers' outputs
d_ff = 2048 # Dimensionality of the inner fully connected layer
n = 6 # Number of layers in the encoder stack

# Define the training parameters
epochs = 2
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1

class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)

        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step_num):
        # Linearly increasing the learning for the first warmup steps and decreasing it there after
        arg1 = step_num ** 0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)

        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)
    
# Instantiate the Adam optimizer
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)

#prepare the training ans testing dataset
dataset = PerpareDateset()

trainX, trainY, valX, valY, train_orig, val_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size  = dataset.call("C:\machine learning\BuildingTransformer\english-german-both.pkl")



# prepare the dataset batches 
train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)

# Prepare the validation dataset batches
val_dataset = data.Dataset.from_tensor_slices((valX, valY))
val_dataset = val_dataset.batch(batch_size)

# # create model
training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)



# defining the loss function 
def loss_fcn(target, prediction):
    # create the mask so that the zero padding values are not included in the computation of the loss
    mask = math.logical_not(equal(target, 0))
    mask = cast(mask, float32)

    # compute the sparse categorical loss on unmasked values 
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * mask

    return reduce_sum(loss) / reduce_sum(mask)

# defining the accuracy function
def accuracy_fcn(target, prediction):
    # create mask so that the zero padding values are not included in the computation of accuracy
    mask = math.logical_not(equal(target, 0))

    # find equal prediction and the target values and apply the padding mask
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(mask, accuracy)

    # Task the true/false values to 32-bit-precision floating point numbers 
    mask = cast(mask, float32)
    accuracy = cast(accuracy, float32)

    # Compute the mean accuracy over the unmasked values 
    return reduce_sum(accuracy) / reduce_sum(mask)

# Include metrics monitoring 
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')
val_loss = Mean(name='val_loss')


# create a checkpoint object and manager to manage multiple checkpoints 
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

# Initialise dictionaries to store the training and validation losses
train_loss_dict = {}
val_loss_dict = {}


# speeding up the training process
@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as Tape:
        # Run the forward pass of the model to generate prediction
        prediction = training_model(encoder_input, decoder_input, training=True)

        # compute the training loss
        loss = loss_fcn(decoder_output, prediction)

        accuracy = accuracy_fcn(decoder_output, prediction)

    # retrive gradients of the trainable variable with respect to training loss
    gradients = Tape.gradient(loss, training_model.trainable_weights)

    # update the values of trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))

    train_loss(loss)
    train_accuracy(accuracy)


for epoch in range(epochs):
    train_loss.reset_state()
    train_accuracy.reset_state()

    print("\nstart of epoch %d" % (epoch + 1))

    start_time = time()

    # Iterate over the dataset batches
    for step, (train_batchX, train_batchY) in enumerate(train_dataset):
        # define the encoder and decoder inputs, and the decoder output
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]

        train_step(encoder_input, decoder_input, decoder_output)

        if step % 50 == 0:
            print(f"Epoch {epoch+1} Step {step} Loss {train_loss.result():.4f} " + f"Accuracy {train_accuracy.result():.4f}")

    # Run a validation step after every epoch of training
    for val_batchX, val_batchY in val_dataset:
        # Define the encoder and decoder inputs, and the decoder output
        encoder_input = val_batchX[:, 1:]
        decoder_input = val_batchY[:, :-1]
        decoder_output = val_batchY[:, 1:]
        # Generate a prediction
        prediction = training_model(encoder_input, decoder_input, training=False)
        # Compute the validation loss
        loss = loss_fcn(decoder_output, prediction)
        val_loss(loss)

    # Print epoch number and accuracy and loss values at the end of every epoch
    print(f"Epoch {epoch+1}: Training Loss {train_loss.result():.4f}, " + f"Training Accuracy {train_accuracy.result():.4f}, "+ f"Validation Loss {val_loss.result():.4f}")
    
    # Save a checkpoint after every epoch
    if (epoch + 1) % 1 == 0:
        save_path = ckpt_manager.save()
        print(f"Saved checkpoint at epoch {epoch+1}")
        # Save the trained model weights
        # training_model.save_weights("weights/wghts" + str(epoch + 1) + ".ckpt")
        training_model.save_weights("weights/wghts" + str(epoch + 1) + ".weights.h5")
        train_loss_dict[epoch] = train_loss.result()

        # training_model.save_weights("weights/wghts" + str(epoch + 1) + ".weights.h5")
        train_loss_dict[epoch] = train_loss.result()
        val_loss_dict[epoch] = val_loss.result()
    
    
# Save the training loss values
with open('./train_loss.pkl', 'wb') as file:
    dump(train_loss_dict, file)

# Save the validation loss values
with open('./val_loss.pkl', 'wb') as file:
    dump(val_loss_dict, file)
print("Total time taken: %.2fs" % (time() - start_time))

