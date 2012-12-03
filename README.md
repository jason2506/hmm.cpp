# HMM.cpp

## Execution

To perform training and testing, just executes:

    $ ./bin/train 100 data/model_init.txt data/seq_model_01.txt model_01.txt 
    $ ./bin/train 100 data/model_init.txt data/seq_model_02.txt model_02.txt 
    $ ./bin/train 100 data/model_init.txt data/seq_model_03.txt model_03.txt 
    $ ./bin/train 100 data/model_init.txt data/seq_model_04.txt model_04.txt 
    $ ./bin/train 100 data/model_init.txt data/seq_model_05.txt model_05.txt 
    $ ./bin/test data/modellist.txt data/testing_data1.txt result1.txt
    $ ./bin/test data/modellist.txt data/testing_data2.txt result2.txt

where `100` is the number of iterations for training.

You can also execute `run.sh` to do the same things:

    $ ./run.sh 100

If you give the path of the answer file as the additional parameter to the test program, it will print the accuracy to the screen:

    $ ./bin/test data/modellist.txt data/testing_data1.txt result1.txt data/testing_answer.txt

