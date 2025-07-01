from utils.dictionary_mnist import n_layer_neurons

def count_total_neurons():
    """
    Calculates and prints the total number of neurons in the network.
    """
    total_neurons = sum(n_layer_neurons)
    print(f"Network Layer Configuration: {n_layer_neurons}")
    print(f"Total number of neurons: {total_neurons}")

if __name__ == "__main__":
    count_total_neurons() 