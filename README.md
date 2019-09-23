Following is the source code for the article "Predicting Performances in Business processes usingDeep Neural Networks" by Gyunam Park and [Minseok Song](http://mssong.postech.ac.kr) submitted to Decision Support Systems.

The code provided in this repository can be readily used to predict the future performances of business processes, given the historical performances using deep neural networks.

### Requirements:

- This code is written in Python3.6.5. In addition, you need to install a few more packages.

  - h5py
  - numpy
  - pandas
  - keras
  - tensorflow
  - scikit-learn
  - scipy
  - PyProM

- To install,

  ```
  $ cd performance_prediction
  $ pip install -r requirements.txt
  $ cd ..
  $ git clone https://github.com/gyunamister/PyProM.git
  ```



### Implementation:

- There are two main files in the source code: prepare_main.py & train_main.py. We divided the main files  to seperate the prepartion for the training sets and the learning.
- "prepare_main.py" is responsible for the preparation of the training sets. It proceeds the construction of annotated process model, the generation of process representation matrix, and the prepartion of the training sets. The resulting training sets are saved into "./matrices/".
  - The keyword arguments are as follows:
    - exp_name: the title of the experiment
    - path: the directory of logs
    - agg: the aggregation function
    - measure: the performance measure (e.g., processing, waiting, sojourn time)
    - fv_format: the format of process representation matrix
    - start: the start date of the log
    - end: the end date of the log
    - range: the range of time window
    - stride: the stride of producing time windows
    - input_size: the length of input time windows
    - output_size: the length of output time windows
    - node_threshold: the threshold for nodes in the process model
    - edge_threshold: the threshold for edges in the process model
    - horizon: the number of prefixes to consider when building transition matrix
- "train_main.py" is responsible for training and testing the approaches presented in the paper (i.e., statistical approach, search-based approach, and deep-learning approach). The results are saved into "./results/".
  - The keyword arguments are as follows:
    - exp_id: the name of the experiment which is given by "prepare.py".
    - search: True if applying search-based approach
    - prediction: state or transition prediction
    - result: the directory of the result file
    - fv_format: the format of process representation matrix
    - algo: name of a model (e.g., CNN, LSTM, LRCN, ...)
- For the detailed implementation, see the main files.

### Evaluation

- Case study 2: BPIC'12
  - The second case study can be reproduced by typing :  `$ sh bpic12_exp_1.sh`
  - The result is reported in `./results/BPIC12_exp_result_1.txt`
- Case study 3: Helpdesk
  - The second case study can be reproduced by typing :  `$ sh hd_exp_1.sh`
  - The result is reported in `./results/HD_exp_result_1.txt`