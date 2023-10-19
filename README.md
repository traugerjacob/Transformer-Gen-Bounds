# Transformer-Gen-Bounds
Code for the paper Sequence Length Independent Generalization Bounds For Transformers

The code given creates a sparse majority dataset and then trains a 1 layer transformer. It prints the normal tensforflow fit output along with the 1 norm of all the weights, training acc/cross entropy, and testing acc/cross entropy. We also save the best weights into a directory given by the sequence length and the 3rd command line argument. 

To run the code, it takes 3 command line arguments: The sequence length size, the batch size, and an integer to denote the run number it is on. An example of it running is:

`run_experiment.py 200 128 1`

This will print what is stated above along with storing the weights using tensorflow's model checkpoint function and then store it in the directory "./200/1/chkpt"
