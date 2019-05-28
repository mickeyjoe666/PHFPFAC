#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "CreateTable/create_PFAC_table_reorder.c"
#include "PHF/phf.c"
#include<pthread.h>



struct thread_data{
    int GPUnum;
    unsigned char *input_string;
    int input_size;
    int state_num;
    int final_state_num;
    unsigned int** match_result;
    int HTSize;
    int width;
    int *s0Table;
    int max_pat_len;
    int* r;
    int* HT;
    int* val;
};


void *GPU_TraceTable(void *threadarg);

/****************************************************************************
*   Function   : main
*   Description: Main function
*   Parameters : Command line arguments
*   Returned   : Program end success(0) or fail(1)
****************************************************************************/
int main(int argc, char *argv[]) {
    //number of GPUs on the machine
    //int GPU_N = 2 ;
    cudaGetDeviceCount(&GPU_N);
    printf("have %d GPU can be used same time\n",GPU_N);
    //Array contaning the number of states in the automaton of each GPU
    int* state_num = (int*)malloc(GPU_N*sizeof(int));
    if(state_num == NULL) {printf("state_num malloc fail\n"); return(1);}
    //Array contaning the number of final states in the automaton of each GPU
    int* final_state_num = (int*)malloc(GPU_N*sizeof(int));
    if(final_state_num == NULL) {printf("final_state_num malloc fail\n"); return(1);}
    //Array contaning maximum pattern length in the automaton of each GPU
    int* max_pat_len_arr = (int*)calloc(GPU_N, sizeof(int));
    if(max_pat_len_arr == NULL) {printf("max_pat_len fail\n"); return(1);}
    //Maximum pattern length over all patterns
    int max_pat_len = 0;
    //Array of the automatons of each GPU
    int*** PFACs = (int***)malloc(GPU_N*sizeof(int**));
    if(PFACs == NULL) {printf("PFACs fail\n"); return(1);}
    //Array of maps from sinal state number to pattern id for each GPU
    int** patternIdMaps = (int**)malloc(GPU_N*sizeof(int*));
    if(patternIdMaps == NULL) {printf("patternIdMaps malloc fail\n"); return(1);}
    //Array contaning the size of the hash table of each GPU
    int* HTSize = (int*)malloc(GPU_N*sizeof(int));
    if(HTSize == NULL) {printf("HTSize malloc fail\n"); return(1);}
    //r[GPU_i][R]=amount row Keys[GPU_i][R][] was shifted
    int** r = (int**)malloc(GPU_N*sizeof(int*));
    if(r == NULL) {printf("r fail\n"); return(1);}
    //the shifted rows of Keys[GPU_i][][] collapse into HT[GPU_i][]
    int** HT = (int**)malloc(GPU_N*sizeof(int*));
    if(HT == NULL) {printf("HT malloc fail\n"); return(1);}
    //store next state corresponding to hash key
    int** val = (int**)malloc(GPU_N*sizeof(int*));
    if(val == NULL) {printf("val malloc fail\n"); return(1);}
    for (int GPUnum = 0; GPUnum < GPU_N; GPUnum++) {
        r[GPUnum] = (int*)malloc((size_t)ROW_MAX*sizeof(int));
	if(r[GPUnum] == NULL) { printf("Failed to malloc r[GPUNum]\n"); return(1);}
        HT[GPUnum] = (int*)malloc((size_t)HASHTABLE_MAX*sizeof(int));
	if(HT[GPUnum] == NULL) { printf("Failed to malloc HT[GPUnum]\n"); return(1);}
        val[GPUnum] = (int*)malloc((size_t)HASHTABLE_MAX*sizeof(int));
	if(val[GPUnum] == NULL) { printf("Failed to malloc val[GPUnum]\n"); return(1);}
	if(r[GPUnum] == NULL || HT[GPUnum] == NULL || val[GPUnum] == NULL) { printf("Problem here\n"); return(1);}
    }
    int type;
    int width; 
    unsigned char *input_string;
    int input_size;
    unsigned int** match_result = (unsigned int**)malloc(GPU_N*sizeof(unsigned int*));
    int i;
    int j;
    int x;



    // check command line arguments
    if (argc != 5) {
        fprintf(stderr, "usage: %s <pattern file name> <type> <PHF width> <input file name>\n", argv[0]);
        exit(-1);
    }

   
    // read pattern file and create PFAC table
    type = atoi(argv[2]);
    //printf("still ok before entry create_PFAC_table_reorder\n");
    create_PFAC_table_reorder(argv[1], state_num, final_state_num, type, max_pat_len_arr, &max_pat_len, PFACs, patternIdMaps);


//    char* fname = "PFAC_table.txt";
//    FILE *fw = fopen(fname, "w");
//    if (fw == NULL) {
//        perror("Open output file failed.\n");
//        exit(1);
//    }
    for(int GPUnum = 0; GPUnum<GPU_N; GPUnum++){
        printf("state num on GPU %d : %d\n", GPUnum, state_num[GPUnum]);
        printf("final state num on GPU %d : %d\n", GPUnum, final_state_num[GPUnum]);
        printf("max pattern length on GPU %d : %d\n", GPUnum, max_pat_len_arr[GPUnum]);

        // output PFAC table
//        fprintf(fw, "PFAC for GPU %d\n", GPUnum);
//        for (i = 0; i < state_num[GPUnum]; i++) {
//            for (j = 0; j < CHAR_SET; j++) {
//                if (PFACs[GPUnum][i][j] != -1) {
//                    fprintf(fw, "state=%2d  '%c'(%02X) ->  %2d\n", i, j, j, PFACs[GPUnum][i][j]);
//                }
//            }
//        }
//        fprintf(fw, "\n\n");
    }
//    fclose(fw);


    // create PHF hash table from PFAC table
    width = atoi(argv[3]);
    for(int GPUnum = 0; GPUnum < GPU_N; GPUnum++){
        HTSize[GPUnum] = FFDM(PFACs[GPUnum], state_num[GPUnum], width, r[GPUnum], HT[GPUnum],val[GPUnum]);
    }

    // read input data
    FILE *fpin = fopen(argv[4], "rb");
    if (fpin == NULL) {
        perror("Open input file failed.");
        exit(1);
    }
    // obtain file size:
    fseek(fpin, 0, SEEK_END);
    input_size = ftell(fpin)-1;
    rewind(fpin);
    printf("input_size here = %d char\n", input_size );

    // allocate host memory: input data
    cudaError_t status;
    status = cudaHostAlloc((void **) &input_string, sizeof(unsigned char)*input_size, cudaHostAllocPortable);
    if (cudaSuccess != status) {
        fprintf(stderr, "cudaMallocHost input_string error: %s\n", cudaGetErrorString(status));
        exit(1);
    }

    // copy the file into the buffer:
    fread(input_string, sizeof(char), input_size, fpin);
    fclose(fpin);


    //TODO: parallelise this. Need to make sure each GPU has its own output variables

    pthread_t threads[GPU_N];
    pthread_attr_t attr;
    int rc;
    void *status1;
    struct thread_data thread_data_array[GPU_N];

    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);



    for (int GPUnum = 0; GPUnum < GPU_N; GPUnum++){



        //Create thread from here
        // exact string matching kernel


        thread_data_array[GPUnum].GPUnum = GPUnum;
        thread_data_array[GPUnum].input_string = input_string;
        thread_data_array[GPUnum].input_size = input_size;
        thread_data_array[GPUnum].state_num = state_num[GPUnum];
        thread_data_array[GPUnum].final_state_num = final_state_num[GPUnum];
        thread_data_array[GPUnum].match_result = match_result;
        thread_data_array[GPUnum].HTSize = HTSize[GPUnum];
        thread_data_array[GPUnum].width = width;
        thread_data_array[GPUnum].s0Table = PFACs[GPUnum][(final_state_num[GPUnum]+1)];
        thread_data_array[GPUnum].max_pat_len =  max_pat_len_arr[GPUnum];
        thread_data_array[GPUnum].r = r[GPUnum];
        thread_data_array[GPUnum].HT = HT[GPUnum];
        thread_data_array[GPUnum].val = val[GPUnum];

        printf("Creating thread %d\n", GPUnum);
        rc = pthread_create(&threads[GPUnum], &attr, GPU_TraceTable, (void *) &thread_data_array[GPUnum]);
        if (rc){
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }


    }

    /* Free attribute and wait for the other threads */
    pthread_attr_destroy(&attr);

    for (int GPUnum = 0; GPUnum < GPU_N; GPUnum++){
        rc = pthread_join(threads[GPUnum], &status1);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
        printf("Completed join with thread %d status1= %ld\n",GPUnum, (long)status1);
    }


    //TODO: synchronise threads that did the matching once the above TODO is done

    cudaFreeHost(input_string);
    printf("cudaFreeHost(input_string); done\n");
    int* match_result_aggreg = (int*)malloc(sizeof(int)*(size_t)input_size*(size_t)max_pat_len);
    memset(match_result_aggreg, 0xFF, sizeof(int)*(size_t)input_size*(size_t)max_pat_len);
    for (int GPUnum = 0; GPUnum < GPU_N; GPUnum++) {
	printf("Matches of GPU %d\n", GPUnum);
        for (i = 0; i < input_size; i++) {
            unsigned int k = (unsigned int)i * (unsigned int)max_pat_len;
            while(match_result_aggreg[k] != -1) k++;
            for (j = 0; j < max_pat_len_arr[GPUnum]; j++) {
                if(match_result[GPUnum][(unsigned int)i*(unsigned int)max_pat_len_arr[GPUnum]+(unsigned int)j] != -1) {
                    int matched_id = patternIdMaps[GPUnum][match_result[GPUnum][(unsigned int)i*(unsigned int)max_pat_len_arr[GPUnum]+(unsigned int)j]];
                    match_result_aggreg[k++] = matched_id;
		    printf("At position %d, match pattern %d\n", i, matched_id);
                    if(matched_id < -1) printf("negative matched id: %d, GPUnum: %d i: %d j: %d\n", matched_id, GPUnum, i, j);
                }
                else
                    break;
            }
        }
        if ( cudaSetDevice(GPUnum) != cudaSuccess ) {
            fprintf(stderr, "Set CUDA device %d error\n", GPUnum);
            exit(1);
        }
        cudaFreeHost(match_result[GPUnum]);
	printf("\n");
    }


    char* output_file_name = "GPU_match_result.txt";
    FILE *fpout1 = fopen(output_file_name, "w");
    if (fpout1 == NULL) {
        perror("Open output file failed.\n");
        exit(1);
    }
    for (i = 0; i < input_size; i++) {
        for (j = 0; j < max_pat_len; j++){
            if (match_result_aggreg[(unsigned int)i*(unsigned int)max_pat_len+(unsigned int)j] != -1) {
                fprintf(fpout1, "At position %4d, match pattern %d\n", i, match_result_aggreg[(unsigned int)i*(unsigned int)max_pat_len+(unsigned int)j]);
                //if(match_result_aggreg[i*max_pat_len+j] < -1) printf("negative matched id: %d, at index %d: i:%d j:%d\n", match_result_aggreg[i*max_pat_len+j], i*max_pat_len+j, i, j);
            } else {
                break;
            }
        }
    }
    fclose(fpout1);
    return 0;

//    pthread_exit(NULL);
}
