#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "CreateTable/create_PFAC_table_reorder.c"
#include "PHF/phf.c"
#include <omp.h>
#include <time.h>


//int num_output[MAX_STATE];            // num of matched pattern for each state
//int *outputs[MAX_STATE];              // list of matched pattern for each state

//int r[ROW_MAX];          // r[R]=amount row Keys[R][] was shifted
//int HT[HASHTABLE_MAX];   // the shifted rows of Keys[][] collapse into HT[]
//int val[HASHTABLE_MAX];  // store next state corresponding to hash key, not used in this version


struct thread_data{
    unsigned char *input_string;
    int input_size;
    int state_num;
    int final_state_num;
    unsigned int* match_result;
    int HTSize;
    int width;
    int *s0Table;
    int max_pat_len;
    int* r;
    int* HT;
    int* val;
};


int GPU_Malloc_Memory(thread_data dataset, unsigned char **d_input_string, int **d_r, int **d_hash_table, unsigned int **d_match_result, int **d_val_table, int **d_s0Table);
int GPU_TraceTable(thread_data dataset, cudaStream_t stream, unsigned char *d_input_string, int *d_r, int *d_hash_table, unsigned int *d_match_result, int *d_val_table, int *d_s0Table);
int GPU_Free_memory(unsigned char **d_input_string, int **d_r, int **d_hash_table, unsigned int **d_match_result, int **d_val_table, int **d_s0Table);

/****************************************************************************
*   Function   : main
*   Description: Main function
*   Parameters : Command line arguments
*   Returned   : Program end success(0) or fail(1)
****************************************************************************/
int main(int argc, char *argv[]) {
    //number of GPUs on the machine
    int streamnum;
    streamnum = atoi(argv[2]);
    int GPU_S;
    cudaGetDeviceCount(&GPU_S);
    int GPU_N = streamnum * GPU_S;
    //Array contaning the number of states in the automaton of each GPU
    int* state_num = (int*)malloc(GPU_N*sizeof(int));
    //Array contaning the number of final states in the automaton of each GPU
    int* final_state_num = (int*)malloc(GPU_N*sizeof(int));
    //Array contaning maximum pattern length in the automaton of each GPU
    int* max_pat_len_arr = (int*)calloc(GPU_N, sizeof(int));
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
    int width;
    unsigned char *input_string;
    int input_size;
    unsigned int** match_result = (unsigned int**)malloc(GPU_N*sizeof(unsigned int*));
    int i;
    int j;
    struct thread_data thread_data_array[GPU_N];
    clock_t start, finish;
    double  mutiGPU_duration;
    size_t size;



    // check command line arguments
    if (argc != 5) {
        fprintf(stderr, "usage: %s <pattern file name> <streamnum> <PHF width> <input file name>\n", argv[0]);
        exit(-1);
    }


    // read pattern file and create PFAC table
    unsigned char *d_input_string[GPU_N];
    int *d_r[GPU_N];
    int *d_hash_table[GPU_N];
    unsigned int *d_match_result[GPU_N];
    int *d_val_table[GPU_N];//add by qiao 0324
    int *d_s0Table[GPU_N];

//    printf("still ok before entry create_PFAC_table_reorder\n");
    create_PFAC_table_reorder(argv[1], state_num, final_state_num, streamnum, max_pat_len_arr, &max_pat_len, PFACs, patternIdMaps);


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

    //TODO: make  hash table created be parallel

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


    // allocate host memory: match_result
    for (int GPUnum = 0; GPUnum < GPU_S; GPUnum++){
        for(int i = 0; i < streamnum; i++){
            int stream_id = GPUnum*streamnum +i;
            status = cudaHostAlloc((void **) &(match_result[stream_id]), sizeof(int)* input_size* max_pat_len_arr[stream_id], cudaHostAllocPortable);
//            printf("using the %d GPU\n", GPUnum);
            if (cudaSuccess != status) {
                fprintf(stderr, "cudaMallocHost match_result error: %s\n", cudaGetErrorString(status));
                exit(1);
            }
        }
    }

    start = clock();


    //create stream for each GPU
    size = GPU_S * sizeof(cudaStream_t);
    stream = (cudaStream_t *)malloc(size);
    cudaStream_t stream[GPU_N];


//    TODO: make muti-GPU things be parallel


    for(int GPUnum = 0; GPUnum < GPU_S; GPUnum++) {

        cudaSetDevice(GPUnum);
        if (cudaSetDevice(GPUnum) != cudaSuccess) {
            fprintf(stderr, "Set CUDA device %d error\n", GPUnum);
            exit(1);
        }
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);//add by qiao 20190402

        for (int i = 0; i < streamnum; i++) {

            int stream_id = GPUnum*streamnum +i;

            thread_data_array[stream_id].input_string = input_string;
            thread_data_array[stream_id].input_size = input_size;
            thread_data_array[stream_id].state_num = state_num[stream_id];
            thread_data_array[stream_id].final_state_num = final_state_num[stream_id];
            thread_data_array[stream_id].match_result = match_result[stream_id];
            thread_data_array[stream_id].HTSize = HTSize[stream_id];
            thread_data_array[stream_id].width = width;
            thread_data_array[stream_id].s0Table = PFACs[stream_id][(final_state_num[stream_id]+1)];
            thread_data_array[stream_id].max_pat_len =  max_pat_len_arr[stream_id];
            thread_data_array[stream_id].r = r[stream_id];
            thread_data_array[stream_id].HT = HT[stream_id];
            thread_data_array[stream_id].val = val[stream_id];

            GPU_Malloc_Memory(thread_data_array[stream_id], &(d_input_string[stream_id]), &(d_r[stream_id]),
                              &(d_hash_table[stream_id]), &(d_match_result[stream_id]), &(d_val_table[stream_id]),
                              &(d_s0Table[stream_id]));

            cudaStreamCreate(&stream[stream_id]);

        }




    }


    omp_set_num_threads(GPU_S);
    #pragma omp parallel for
    for(int GPUnum = 0; GPUnum < GPU_S; GPUnum++) {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        int gpu_id = -1;

        cudaSetDevice(cpu_thread_id%GPU_S);
        if ( cudaSetDevice(cpu_thread_id%GPU_S) != cudaSuccess ) {
            fprintf(stderr, "Set CUDA device %d error\n", cpu_thread_id%GPU_S);
            exit(1);
        }
        cudaGetDevice(&gpu_id);
        if ( cudaGetDevice(&gpu_id) != cudaSuccess ) {
            fprintf(stderr, "Get CUDA device %d error\n", gpu_id);
            exit(1);
        }

        printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);

//        // record time setting
//        cudaEvent_t start, stop;
//        float time;
//        cudaEventCreate(&start);
//        cudaEventCreate(&stop);
//        cudaEventRecord(start, 0);

//        omp_set_num_threads(streamnum);
//        #pragma omp parallel for
        for(int i = 0; i < streamnum; i++){
            unsigned int stream_thread_id = omp_get_thread_num();
            int stream_id = GPUnum*streamnum +stream_thread_id;
            printf("stream is %d now \n",i);
            GPU_TraceTable(thread_data_array[stream_id], stream[stream_id], d_input_string[stream_id], d_r[stream_id], d_hash_table[stream_id], d_match_result[stream_id], d_val_table[stream_id], d_s0Table[stream_id]);
        }



//        // record time setting
//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&time, start, stop);
//
//        printf("The GPU elapsed time is %f ms\n", time);
    }





    for(int GPUnum = 0; GPUnum < GPU_S; GPUnum++) {
        for(int i = 0; i < streamnum; i++){
            int stream_id = GPUnum * streamnum + i;
            cudaSetDevice(GPUnum);
            if (cudaSetDevice(GPUnum) != cudaSuccess) {
                fprintf(stderr, "Set CUDA device %d error\n", GPUnum);
                exit(1);
            }
            GPU_Free_memory(&(d_input_string[stream_id]), &(d_r[stream_id]), &(d_hash_table[stream_id]), &(d_match_result[stream_id]), &(d_val_table[stream_id]), &(d_s0Table[stream_id]));

            cudaStreamDestroy(stream[i]);
        }
    }

    printf("/////////////////////////////////////////////\n");

















    finish = clock();
    mutiGPU_duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf( "Time for  %d GPU is %f seconds\n", GPU_S, mutiGPU_duration );




    //TODO: synchronise threads that did the matching once the above TODO is done
    cudaFreeHost(input_string);
    printf("cudaFreeHost(input_string); done\n");
    printf("max_pat_len is %d \n",max_pat_len);



//    for(int GPUnum = 0; GPUnum < GPU_N; GPUnum++){
//        for(int i = 0; i< final_state_num[GPUnum];i++){
//            printf("pattern id is %d\n", patternIdMaps[GPUnum][i]);
//
//        }
//    }

//    printf("x is %d\n", x);


    int* match_result_aggreg = (int*)malloc(sizeof(int)*(size_t)input_size*(size_t)max_pat_len);
    memset(match_result_aggreg, 0xFF, sizeof(int)*(size_t)input_size*(size_t)max_pat_len);
    for (int GPUnum = 0; GPUnum < GPU_N; GPUnum++) {
        for (i = 0; i < input_size; i++) {
            unsigned int k = (unsigned int)i * (unsigned int)max_pat_len;
            while(match_result_aggreg[k] != -1) {
//                printf("match_result_aggreg[k] is %d, k is%d \n", match_result_aggreg[k], k);
                k++;
            }
//            printf("the problem comes from here\n");
            for (j = 0; j < max_pat_len_arr[GPUnum]; j++) {
                if(match_result[GPUnum][(unsigned int)i*(unsigned int)max_pat_len_arr[GPUnum]+(unsigned int)j] != -1) {
                    int matched_id = patternIdMaps[GPUnum][match_result[GPUnum][(unsigned int)i*(unsigned int)max_pat_len_arr[GPUnum]+(unsigned int)j]];
                    match_result_aggreg[k++] = matched_id;
//                    printf("At position %d, match pattern %d,match result is %d\n", i, matched_id, match_result[GPUnum][(unsigned int)i*(unsigned int)max_pat_len_arr[GPUnum]+(unsigned int)j]);
                    if(matched_id < -1) printf("negative matched id: %d, GPUnum: %d i: %d j: %d\n", matched_id, GPUnum, i, j);
                }
                else
                    break;
            }
//            printf("k is %d\n",k);
        }
        cudaFreeHost(match_result[GPUnum]);
        printf("match_result memory is freed, %d\n",GPUnum);
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
