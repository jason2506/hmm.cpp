CC=g++
CFLAGS=-c -Wall -O3
LDFLAGS=
TARGET=../bin

all: train test

train: train.o hmm.o
	mkdir -p ${TARGET}
	${CC} hmm.o train.o -o ${TARGET}/train

test: test.o hmm.o
	mkdir -p ${TARGET}
	${CC} hmm.o test.o -o ${TARGET}/test

train.o: train.cpp
	$(CC) $(CFLAGS) train.cpp

test.o: test.cpp
	$(CC) $(CFLAGS) test.cpp

hmm.o:  hmm.h hmm.cpp
	$(CC) $(CFLAGS) hmm.cpp

clean:
	$(RM) *.o
	$(RM) -r ${TARGET}

