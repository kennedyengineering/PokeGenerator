from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm, trange


def build_reverse_process_mlp_model(input_dim, num_layers, num_hidden, T):
    """ Builds the reverse process MLP model using latent vectors from an autoencoder.

    Arguments:
        input_dim: dimension of the latent vectors
        num_layers: number of layers in the MLP
        num_hidden: number of neurons in each hidden layer
        T: maximum timestep, determines the size of the embedding layer

    Returns:
        Keras model
    """
    # Latent vector input (from autoencoder)
    latent_input = layers.Input(shape=(input_dim,))

    # Timestep input
    timestep_input = layers.Input(shape=(1,))

    # Embedding layer for timestep, mapped to the size of the hidden layers
    embedding = layers.Embedding(T, num_hidden, embeddings_initializer='glorot_normal')(timestep_input)
    embedding = layers.Flatten()(embedding)  # Flatten embedding output to match latent vector dimensions

    # Concatenate embedding vector to the latent vector input
    concatenated = layers.Concatenate(axis=-1)([latent_input, embedding])

    # Process concatenated inputs through the MLP
    x = concatenated
    for _ in range(num_layers):
        x = layers.Dense(num_hidden, activation='relu')(x)

    # Output layer: output a vector matching the latent vector dimension
    output = layers.Dense(input_dim, activation=None)(x)

    model = keras.Model(inputs=[latent_input, timestep_input], outputs=output, name='reverse_process_mlp_model')
    return model

def prepare_dataset(latent_vectors, batch_size):
    """Prepares the training dataset from latent vectors."""
    dataset = tf.data.Dataset.from_tensor_slices(latent_vectors)
    return dataset.shuffle(buffer_size=1024).batch(batch_size)

def generate_training_batch(x_batch_train, T, alphas_cumprod):
    """Generates a training batch with corresponding noise and timesteps."""
    # Generate a batch of timesteps
    t = np.random.randint(T, size=(len(x_batch_train),1))

    # Generate noise
    noise = np.random.normal(size=x_batch_train.shape)

    # Calculate at for each timestep in the batch
    at = alphas_cumprod[t]
    at = np.reshape(at,(-1,1))

    # Calculate inputs considering the shape of x_batch_train is (batch_size, 512)
    inputs = np.sqrt(at) * x_batch_train + (1 - at) * noise

    return inputs, noise, t

def train_epoch(dataset, model, optimizer, loss_fn, T, alphas_cumprod):
    """Trains the model for one epoch."""
    total_loss = 0
    for x_batch_train in dataset:
        inputs, noise, t = generate_training_batch(x_batch_train, T, alphas_cumprod)
        with tf.GradientTape() as tape:
            est_noise = model([inputs, t])
            loss_value = loss_fn(noise, est_noise)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        total_loss += loss_value.numpy()
    return total_loss / len(dataset)

def train_model(latent_vectors, batch_size, T, alphas_cumprod, model, epochs=100):
    """Trains the latent diffusion model."""
    train_dataset = prepare_dataset(latent_vectors, batch_size)
    optimizer = keras.optimizers.Adam(3e-4)
    loss_fn = keras.losses.MeanSquaredError()

    for epoch in range(epochs):
        total_loss = train_epoch(train_dataset, model, optimizer, loss_fn, T, alphas_cumprod)
        print(f'Loss at epoch {epoch}: {total_loss}')
    return model

def sample(model, shape, T, sigmas, alphas, alphas_cumprod):
  """ Samples from the diffusion model.

  Arguments:
    model: reverse process model
    shape: shape of data to be sampled; should be [N,H,W,1]
  Returns:
    Sampled images
  """
  # sample normally-distributed random noise (x_T)
  x = np.random.normal(size=shape)

  # iterate through timesteps from T-1 to 0
  for t in trange(T-1,-1,-1):
    # sample noise unless at final step (which is deterministic)
    z = np.random.normal(size=shape) if t > 0 else np.zeros(shape)

    # estimate correction using model conditioned on timestep
    eps = model.predict([x,np.ones((shape[0],1))*t],verbose=False)

    # apply update formula
    sigma = sigmas[t]
    a = alphas[t]
    a_bar = alphas_cumprod[t]
    x = 1/np.sqrt(a)*(x - (1-a)/np.sqrt(1-a_bar)*eps)+sigma*z
  return x

# # Testing Creating the Model
# model = build_reverse_process_mlp_model(512, 3, 256, 1000)
# model.summary()