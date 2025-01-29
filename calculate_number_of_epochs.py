import math


if __name__ == "__main__":

    epochs=10

    dataset_size = 7000
    bs = 16
    gradient_acc_steps = 8


    if epochs is None:
        iter_number = 7000
        batch_size = bs * gradient_acc_steps
        epochs = math.ceil(iter_number / (dataset_size // batch_size))
        print(f'Epochs for {iter_number} iters: ', epochs)
    else:
        batch_size = bs * gradient_acc_steps
        num_update_steps_per_epoch = dataset_size // batch_size
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        # we need it for correct calculation of warmup steps
        total_steps = math.ceil(epochs * num_update_steps_per_epoch)
        print(f"Iters for {epochs} epochs: ", total_steps)