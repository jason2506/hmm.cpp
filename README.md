# HMM.cpp

## Execution

To perform training and testing, just executes:

    $ ./train 100 model_init.txt seq_model_01.txt model_01.txt 
    $ ./train 100 model_init.txt seq_model_02.txt model_02.txt 
    $ ./train 100 model_init.txt seq_model_03.txt model_03.txt 
    $ ./train 100 model_init.txt seq_model_04.txt model_04.txt 
    $ ./train 100 model_init.txt seq_model_05.txt model_05.txt 
    $ ./test modellist.txt testing_data1.txt result1.txt
    $ ./test modellist.txt testing_data2.txt result2.txt

where `100` is the number of iterations for training.

You can also execute `run.sh` to do the same things:

    $ ./run.sh 100

If you give the path of the answer file as the additional parameter to the test program, it will print the accuracy to the screen:

    $ ./test modellist.txt testing_data1.txt result1.txt testing_answer.txt

