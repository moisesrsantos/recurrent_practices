library(tensorflow)
library(ggplot2)
library(keras)
source("sequence_gen.R")

data = scan("stockmarket_f.data")
data
data = data.matrix(data)

train_data = data[1:2000]
mean = mean(train_data)
std = sd(train_data)
data = scale(data, center = mean, scale = std)


lookback = 20
step = 1
delay = 1
batch_size = 128

train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 1500,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1501,
  max_index = 2000,
  step = step,
  batch_size = batch_size
)

test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 2001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

val_steps <- (2000 - 1501 - lookback) / batch_size

test_steps <- (nrow(data) - 2001 - lookback) / batch_size


model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(lookback / step, dim(data))) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 20,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)
