Flow of code:
step-1:
	- Create the data loaders of the input data.
	- Each dataset has diffrent way of loading, thus dataset.py has dataset classes to help loading data.
Step-2:
	- Partition the data among the parties. The number of parties are input from user.
	- The data is partitioned based on Derichilet distribution with beta = 0.5
Step-3:
	- If the training method is local-training then run the training without ditributing the dataset.
	- If the training algorithm is MOON, FedProx or FedAvg run corresponding training steps.
Step-4
	- Run sequentially train all the model on all the parties sequentially
	- Once epochs are over per party, then club the results and report.
