#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "CreateTable/create_PFAC_table_reorder.c"
#include "PHF/phf.c"
#include<pthread.h>

int num_output[MAX_STATE];            // num of matched pattern for each state
int *outputs[MAX_STATE];              // list of matched pattern for each state

//int r[ROW_MAX];          // r[R]=amount row Keys[R][] was shifted
//int HT[HASHTABLE_MAX];   // the shifted rows of Keys[][] collapse into HT[]
//int val[HASHTABLE_MAX];  // store next state corresponding to hash key, not used in this version


struct thread_data{ 
    int GPUnum;
    unsigned char *input_string;
    int input_size;
    int state_num;
    int final_state_num;
    unsigned int* match_result;
    int HTSize;
    int width;
    int *s0Table;
    int max_pat_len;
    int r[];
    int HT[];
    int val[];
}; 


struct thread_data thread_data_array[GPUnum];

int GPU_TraceTable(void *threadarg);

/****************************************************************************
*   Function   : main
*   Description: Main function
*   Parameters : Command line arguments
*   Returned   : Program end success(0) or fail(1)
****************************************************************************/
int main(int argc, char *argv[]) {
    //number of GPUs on the machine
    int GPU_N = 2 ;
    //cudaGetDeviceCount(&GPU_N);
    //Array contaning the number of states in the automaton of each GPU
    int* state_num = (int*)malloc(GPU_N*sizeof(int));
    //Array contaning the number of final states in the automaton of each GPU
    int* final_state_num = (int*)malloc(GPU_N*sizeof(int));
    //Array contaning maximum pattern length in the automaton of each GPU
    int* max_pat_len_arr = (int*)malloc(GPU_N*sizeof(int));
    //Maximum pattern length over all patterns
    int max_pat_len = 0;
    //Array of the automatons of each GPU
    int*** PFACs = (int***)malloc(GPU_N*sizeof(int**));
    //Array of maps from sinal state number to pattern id for each GPU
    int** patternIdMaps = (int**)malloc(GPU_N*sizeof(int*));
    //Array contaning the size of the hash table of each GPU
    int* HTSize = (int*)malloc(GPU_N*sizeof(int));
    //r[GPU_i][R]=amount row Keys[GPU_i][R][] was shifted
    int** r = (int**)malloc(GPU_N*sizeof(int*));
    //the shifted rows of Keys[GPU_i][][] collapse into HT[GPU_i][]
    int** HT = (int**)malloc(GPU_N*sizeof(int*));
    //store next state corresponding to hash key
    int** val = (int**)malloc(GPU_N*sizeof(int*));
    for (int GPUnum = 0; GPUnum < GPU_N; GPUnum++) {
        r[GPUnum] = (int*)malloc(ROW_MAX*sizeof(int));
        HT[GPUnum] = (int*)malloc(HASHTABLE_MAX*sizeof(int));
        val[GPUnum] = (int*)malloc(HASHTABLE_MAX*sizeof(int));
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
    printf("still ok before entry create_PFAC_table_reorder\n");
    create_PFAC_table_reorder(argv[1], state_num, final_state_num, type, max_pat_len_arr, &max_pat_len, PFACs, patternIdMaps);


    char* fname = "PFAC_table.txt";
    FILE *fw = fopen(fname, "w");
    if (fw == NULL) {
        perror("Open output file failed.\n");
        exit(1);
    }
    for(int GPUnum = 0; GPUnum<GPU_N; GPUnum++){
        printf("state num on GPU %d : %d\n", GPUnum, state_num[GPUnum]);
        printf("final state num on GPU %d : %d\n", GPUnum, final_state_num[GPUnum]);
        printf("max pattern length on GPU %d : %d\n", GPUnum, max_pat_len_arr[GPUnum]);

        // output PFAC table
        fprintf(fw, "PFAC for GPU %d\n", GPUnum);
        for (i = 0; i < state_num[GPUnum]; i++) {
            for (j = 0; j < CHAR_SET; j++) {
                if (PFACs[GPUnum][i][j] != -1) {
                    fprintf(fw, "state=%2d  '%c'(%02X) ->  %2d\n", i, j, j, PFACs[GPUnum][i][j]);
                }
            }
        }
        fprintf(fw, "\n\n");
    }
    fclose(fw);


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

    // allocate host memory: input data
    cudaError_t status;
    //status = cudaMallocHost((void **) &input_string, sizeof(char)*input_size);
    status = cudaHostAlloc((void **) &input_string, sizeof(char)*input_size, cudaHostAllocPortable);
    if (cudaSuccess != status) {
        fprintf(stderr, "cudaMallocHost input_string error: %s\n", cudaGetErrorString(status));
        exit(1);
    }

    // copy the file into the buffer:
    fread(input_string, sizeof(char), input_size, fpin);
    fclose(fpin);


    //TODO: parallelise this. Need to make sure each GPU has its own output variables
    for(int GPUnum = 0; GPUnum < GPU_N; GPUnum++){

        //Create thread from here
        int rc;
        // exact string matching kernel
        pthread_t [GPUnum];

        thread_data_array[GPUnum].GPUnum = GPUnum;
        thread_data_array[GPUnum].input_string = input_string;
        thread_data_array[GPUnum].input_size = input_size;
        thread_data_array[GPUnum].state_num = state_num[GPUnum];
        thread_data_array[GPUnum].final_state_num = final_state_num[GPUnum];
        thread_data_array[GPUnum].match_result = match_result[GPUnum];
        thread_data_array[GPUnum].HTSize = HTSize[GPUnum];
        thread_data_array[GPUnum].width = width;
        thread_data_array[GPUnum].s0Table = PFACs[GPUnum][(final_state_num[GPUnum]+1)];
        thread_data_array[GPUnum].max_pat_len =  max_pat_len_arr[GPUnum];
        thread_data_array[GPUnum].r = r[GPUnum];
        thread_data_array[GPUnum].HT = HT[GPUnum];
        thread_data_array[GPUnum].val = val[GPUnum];


        rc = int pthread_create(&GPUnum, NULL, GPU_TraceTable, (void *) &thread_data_array[GPUnum]);
        if (rc){
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);

        }

        pthread_exit(NULL);
        // Output results
        //char output_file_name[100] = "GPU_match_result";
        //char number[10];
        //sprintf(number, "%d" , GPUnum);
        //strcat(output_file_name, number);
        //strcat(output_file_name, ".txt");

        //FILE *fpout1 = fopen(output_file_name, "w");
        //if (fpout1 == NULL) {
        //    perror("Open output file failed.\n");
        //    exit(1);
        //}

        // Output match result to file
        //if (type == 0){
        //    for (i = 0; i < input_size; i++) {
        //        for (j = 0; j < max_pat_len_arr[GPUnum]; j++){
        //            if(match_result[GPUnum][i*max_pat_len_arr[GPUnum]+j] != -1) {
        //                int matched_id = patternIdMaps[GPUnum][match_result[GPUnum][i*max_pat_len_arr[GPUnum]+j]];
        //                fprintf(fpout1, "At position %4d, match pattern %d\n", i, matched_id);
        //            }
        //        }
        //    }
        //}
        //fclose(fpout1);
        cudaError_t cuda_err;
        cuda_err = cudaGetLastError() ;
        if ( cudaSuccess != cuda_err ) {
            printf("after the call of kernel function once: error = %s\n", cudaGetErrorString (cuda_err));
            exit(1) ;
        }
    }

    //TODO: synchronise threads that did the matching once the above TODO is done
    cudaFreeHost(input_string);
    printf("cudaFreeHost(input_string); done\n");

    int* match_result_aggreg = (int*)malloc(sizeof(int)*(size_t)input_size*(size_t)max_pat_len);
    memset(match_result_aggreg, 0xFF, sizeof(int)*(size_t)input_size*(size_t)max_pat_len);
    for (int GPUnum = 0; GPUnum < GPU_N; GPUnum++) {
        for (i = 0; i < input_size; i++) {
            unsigned int k = (unsigned int)i * (unsigned int)max_pat_len;
            while(match_result_aggreg[k] != -1) k++;
            for (j = 0; j < max_pat_len_arr[GPUnum]; j++) {
                if(match_result[GPUnum][(unsigned int)i*(unsigned int)max_pat_len_arr[GPUnum]+(unsigned int)j] != -1) {
                    int matched_id = patternIdMaps[GPUnum][match_result[GPUnum][(unsigned int)i*(unsigned int)max_pat_len_arr[GPUnum]+(unsigned int)j]];
                    match_result_aggreg[k++] = matched_id;
                    if(matched_id < -1) printf("negative matched id: %d, GPUnum: %d i: %d j: %d\n", matched_id, GPUnum, i, j);
                }
                else
                    break;
            }
        }
        cudaFreeHost(match_result[GPUnum]);
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
}
