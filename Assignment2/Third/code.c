/*Odd-Even Sort Algorithm
1.Odd Phase: Compare and swap elements at odd indices with their right neighbors.
2.Even Phase: Compare and swap elements at even indices with their right neighbors.
3.Repeat the above phases until the array is sorted.*/


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h> 

int IncOrder(const void *e1, const void *e2);
void CompareSplit(int nlocal, int *elmnts, int *relmnts, int *wspace, int keepsmall);

int main(int argc, char *argv[]) {
    int n;              
    int npes;           
    int myrank;        
    int nlocal;        
    int *elmnts;        
    int *relmnts;       
    int oddrank;        
    int evenrank;       
    int *wspace;        
    int i;
    MPI_Status status;

    /* Initialize MPI and get system information */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    /* Get total number of elements from command line */
    if (argc != 2) {
        if (myrank == 0) {
            fprintf(stderr, "Usage: %s <number of elements>\n", argv[0]);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    n = atoi(argv[1]);
    nlocal = n / npes; 

    /* Allocate memory for the arrays */
    elmnts  = (int *)malloc(nlocal * sizeof(int));
    relmnts = (int *)malloc(nlocal * sizeof(int));
    wspace  = (int *)malloc(nlocal * sizeof(int));

    /* Fill-in the elmnts array with random elements */
    srandom(myrank);
    for (i = 0; i < nlocal; i++) {
        elmnts[i] = random() % 100;
    }

    /* Sort the local elements using the built-in quicksort routine */
    qsort(elmnts, nlocal, sizeof(int), IncOrder);

    
    if (myrank % 2 == 0) {
        oddrank  = myrank - 1;
        evenrank = myrank + 1;
    } else {
        oddrank  = myrank + 1;
        evenrank = myrank - 1;
    }

    
    if (oddrank == -1 || oddrank == npes) 
        oddrank = MPI_PROC_NULL;
    if (evenrank == -1 || evenrank == npes) 
        evenrank = MPI_PROC_NULL;


    for (i = 0; i < npes - 1; i++) {
        if (i % 2 == 1) { /* Odd phase */
            MPI_Sendrecv(elmnts, nlocal, MPI_INT, oddrank, 1, 
                         relmnts, nlocal, MPI_INT, oddrank, 1, 
                         MPI_COMM_WORLD, &status);
        } else {
            /* Even phase */
            MPI_Sendrecv(elmnts, nlocal, MPI_INT, evenrank, 1, 
                         relmnts, nlocal, MPI_INT, evenrank, 1, 
                         MPI_COMM_WORLD, &status);
        }

        /* Perform Compare-Split operation */
        CompareSplit(nlocal, elmnts, relmnts, wspace, myrank < status.MPI_SOURCE);
    }

    /* Print sorted elements in rank order */
    for (i = 0; i < npes; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (myrank == i) {
            printf("Rank %d sorted elements: ", myrank);
            for (int j = 0; j < nlocal; j++) {
                printf("%d ", elmnts[j]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(elmnts);
    free(relmnts);
    free(wspace);

    MPI_Finalize();
    return 0;
}


void CompareSplit(int nlocal, int *elmnts, int *relmnts, int *wspace, int keepsmall) {
    int i, j, k;

   
    for (i = 0; i < nlocal; i++) {
        wspace[i] = elmnts[i];
    }

    if (keepsmall) { 
        for (i = j = k = 0; k < nlocal; k++) {
            if (j == nlocal || (i < nlocal && wspace[i] < relmnts[j])) {
                elmnts[k] = wspace[i++];
            } else {
                elmnts[k] = relmnts[j++];
            }
        }
    } else { 
        for (i = k = nlocal - 1, j = nlocal - 1; k >= 0; k--) {
            if (j < 0 || (i >= 0 && wspace[i] >= relmnts[j])) {
                elmnts[k] = wspace[i--];
            } else {
                elmnts[k] = relmnts[j--];
            }
        }
    }
}


int IncOrder(const void *e1, const void *e2) {
    return (*(int *)e1 - *(int *)e2);
}

