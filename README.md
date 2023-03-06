# Fast Offline Policy Optimization for Large Scale Recommendation

Source code for the paper "Fast Offline Policy Optimization for Large Scale Recommendation" published at the Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI-23).

## Creating the environment

A Conda virtual Python environment can be created from the env.yml file holding all the needed packages to reproduce the results of the paper.

    conda env create -f env.yml
    conda activate fopo
    
## Datasets

The experiments in the paper use the Twitch and Goodreads datasets. They should be downloaded, extract their csv files and put in the preprocessing folder. The csv files are then transformed to sparse datasets by running the following scripts:

    python goodreads_to_sparse.py
    python twitch_to_sparse.py

Once you generate the sparse npz files, you can proceed at setting up the experiments.

## Experiments

### Setup Experiments

This step creates the different splits of the datasets (Train, Validation and Test), reduces the dimension to create product embeddings, create an index on the item embeddings then saves everything into a setup folder. The setups used for the datasets are created by running the following codes:

Twitch:

    python setup_experiment.py --dataset twitch --N_sub 500000 --K 1000
    
GoodReads:

    python setup_experiment.py --dataset goodreads --N_sub 300000 --K 1000
    
### Running Experiments

Once the setup folder is created, we can run an experiment to test a method in {exact, uniform, mixture} for the specific setup. Given the <folder path>, we can for example test exact reinforce by running the following script:
    
    python launch_experiment.py --exp_folder <folder path> --method exact
    
All the results of the experiment are then stored in the same <folder path>.
    

    
    
    
    





