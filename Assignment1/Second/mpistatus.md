# MPI Status Object Example

## Description
This program demonstrates the use of the `MPI_Status` object in an `MPI_Recv` call to determine the number of elements received in a message. The program consists of two processes: one process randomly determines the number of integers to send, and the other process receives the message and checks how many numbers were actually received.

## How It Works
1. **Initialize MPI**: The MPI environment is initialized using `MPI_Init`.
2. **Check Number of Processes**: The program ensures that exactly two processes are being used; otherwise, it aborts.
3. **Random Number Generation**:
   - If the process rank is 0, it generates a random number of integers to send.
   - The generated numbers are sent using `MPI_Send` to process 1.
4. **Receiving Process**:
   - Process 1 receives the message using `MPI_Recv`.
   - The `MPI_Status` object is checked to determine the actual number of elements received.
   - The received count, along with the message source and tag, is printed.
5. **Synchronization and Finalization**:
   - The program uses `MPI_Barrier` to synchronize processes.
   - The MPI environment is finalized using `MPI_Finalize`.

## Syntax of `MPI_Recv`
### MPI_Recv
```c
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)

- `buf`: Pointer to the buffer where received data is stored.
- `count`: Maximum number of elements to receive.
- `datatype`: Data type of elements being received.
- `source`: Rank of the sender process (or `MPI_ANY_SOURCE`).
- `tag`: Message tag (or `MPI_ANY_TAG`).
- `comm`: Communicator (typically `MPI_COMM_WORLD`).
- `status`: MPI_Status object containing information about the received message.
```


