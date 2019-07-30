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
    int GPU_N;
    cudaGetDeviceCount(&GPU_N);
    int stream_N = streamnum * GPU_N;
    //Array contaning the number of states in the automaton of each GPU
    int* state_num = (int*)malloc(stream_N*sizeof(int));
    //Array contaning the number of final states in the automaton of each GPU
    int* final_state_num = (int*)malloc(stream_N*sizeof(int));
    //Array contaning maximum pattern length in the automaton of each GPU
    int* max_pat_len_arr = (int*)calloc(stream_N, sizeof(int));
    //Maximum pattern length over all patterns
    int max_pat_len = 0;
    //Array of the automatons of each GPU
    int*** PFACs = (int***)malloc(stream_N*sizeof(int**));
    //Array of maps from sinal state number to pattern id for each GPU
    int** patternIdMaps = (int**)malloc(stream_N*sizeof(int*));
    //Array contaning the size of the hash table of each GPU
    int* HTSize = (int*)malloc(stream_N*sizeof(int));
    //r[GPU_i][R]=amount row Keys[GPU_i][R][] was shifted
    int** r = (int**)malloc(stream_N*sizeof(int*));
    //the shifted rows of Keys[GPU_i][][] collapse into HT[GPU_i][]
    int** HT = (int**)malloc(stream_N*sizeof(int*));
    //store next state corresponding to hash key
    int** val = (int**)malloc(stream_N*sizeof(int*));
    for (int GPUnum = 0; GPUnum < stream_N; GPUnum++) {
        r[GPUnum] = (int*)malloc(ROW_MAX*sizeof(int));
        HT[GPUnum] = (int*)malloc(HASHTABLE_MAX*sizeof(int));
        val[GPUnum] = (int*)malloc(HASHTABLE_MAX*sizeof(int));
    }
    int width;
    unsigned char *input_string;
    int input_size;
    unsigned int** match_result = (unsigned int**)malloc(stream_N*sizeof(unsigned int*));
    int i;
    int j;
    struct thread_data thread_data_array[stream_N];
    double start_PFAC, finish_PFAC, start_Hashtable, finish_Hashtable, start_multiGPU, finish_multiGPU, start_mallocGPU, finish_mallocGPU;
    double  PFAC_duration, Hashtable_duration,mallocGPU_duration, mutiGPU_duration;


    struct timespec test_b, test_e, test_b1, test_e1, test_b2, test_e2;
    double tesTime, tesTime1, tesTime2;

    
    // check command line arguments
    if (argc != 5) {
        fprintf(stderr, "usage: %s <pattern file name> <streamnum> <PHF width> <input file name>\n", argv[0]);
        exit(-1);
    }

    // read pattern file and create PFAC table
    unsigned char *d_input_string[stream_N];
    int *d_r[stream_N];
    int *d_hash_table[stream_N];
    unsigned int *d_match_result[stream_N];
    int *d_val_table[stream_N];//add by qiao 0324
    int *d_s0Table[stream_N];

    //create PFAC tables
    start_PFAC = omp_get_wtime();
    create_PFAC_table_reorder(argv[1], state_num, final_state_num, streamnum, max_pat_len_arr, &max_pat_len, PFACs, patternIdMaps);
    finish_PFAC = omp_get_wtime();
    PFAC_duration = (double)(finish_PFAC - start_PFAC) ;


    for(int GPUnum = 0; GPUnum<stream_N; GPUnum++){
        printf("state num on GPU %d : %d\n", GPUnum, state_num[GPUnum]);
        printf("final state num on GPU %d : %d\n", GPUnum, final_state_num[GPUnum]);
        printf("max pattern length on GPU %d : %d\n", GPUnum, max_pat_len_arr[GPUnum]);
    }

    // create PHF hash table from PFAC table
    width = atoi(argv[3]);
    start_Hashtable = omp_get_wtime();
    omp_set_num_threads(stream_N);
    #pragma omp parallel for
    for(int GPUnum = 0; GPUnum < stream_N; GPUnum++){
        HTSize[GPUnum] = FFDM(PFACs[GPUnum], state_num[GPUnum], width, r[GPUnum], HT[GPUnum],val[GPUnum]);
    }
    finish_Hashtable = omp_get_wtime();
    Hashtable_duration = (double)(finish_Hashtable - start_Hashtable);

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
    printf("input size is %d char\n", input_size);



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
    for (int GPUnum = 0; GPUnum < GPU_N; GPUnum++){
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


    //start record time
    clock_gettime( CLOCK_REALTIME, &test_b);

    clock_gettime( CLOCK_REALTIME, &test_b1);
    //create stream for each GPU
    cudaStream_t stream[stream_N];

    start_mallocGPU = omp_get_wtime();

    omp_set_num_threads(GPU_N);
    #pragma omp parallel for
    for(int GPUnum = 0; GPUnum < GPU_N; GPUnum++) {
        cudaSetDevice(GPUnum);
        if (cudaSetDevice(GPUnum) != cudaSuccess) {
            fprintf(stderr, "Set CUDA device %d error\n", GPUnum);
            exit(1);
        }
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);//add by qiao 20190402

        clock_gettime( CLOCK_REALTIME, &test_b2);
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
        clock_gettime( CLOCK_REALTIME, &test_e2);
        tesTime2 = (test_e2.tv_sec - test_b2.tv_sec) * 1000.0;
        tesTime2 += (test_e2.tv_nsec - test_b2.tv_nsec) / 1000000.0;
        printf("time for malloc in loop: %lf ms\n", tesTime2);

    }
    finish_mallocGPU = omp_get_wtime();
    mallocGPU_duration = (double)(finish_mallocGPU - start_mallocGPU) ;



    start_multiGPU = omp_get_wtime();
    //start multiGPU thread
    omp_set_num_threads(stream_N);
    #pragma omp parallel for
    for(int GPUnum = 0; GPUnum < stream_N; GPUnum++) {
        int gpu_id = -1;
        if ( cudaSetDevice(GPUnum/streamnum) != cudaSuccess ) {
            fprintf(stderr, "Set CUDA device %d error\n", GPUnum/streamnum);
            exit(1);
        }
        if ( cudaGetDevice(&gpu_id) != cudaSuccess ) {
            fprintf(stderr, "Get CUDA device %d error\n", gpu_id);
            exit(1);
        }

//        printf("CPU thread %d (of %d) uses CUDA device %d\n", GPUnum, stream_N, gpu_id);
            GPU_TraceTable(thread_data_array[GPUnum], stream[GPUnum], d_input_string[GPUnum], d_r[GPUnum], d_hash_table[GPUnum], d_match_result[GPUnum], d_val_table[GPUnum], d_s0Table[GPUnum]);

    }
    finish_multiGPU = omp_get_wtime();
    mutiGPU_duration = (double)(finish_multiGPU - start_multiGPU) ;

    clock_gettime( CLOCK_REALTIME, &test_e);
    tesTime = (test_e.tv_sec - test_b.tv_sec) * 1000.0;
    tesTime += (test_e.tv_nsec - test_b.tv_nsec) / 1000000.0;
    printf("time before free memory: %lf ms\n", tesTime);


    clock_gettime( CLOCK_REALTIME, &test_b);

    //synchronise threads and free memory on GPU
    for(int GPUnum = 0; GPUnum < GPU_N; GPUnum++) {
        for(int i = 0; i < streamnum; i++){
            int stream_id = GPUnum * streamnum + i;
            cudaError_t cuda_err;
            cudaSetDevice(GPUnum);
            if (cudaSetDevice(GPUnum) != cudaSuccess) {
                fprintf(stderr, "Set CUDA device %d error\n", GPUnum);
                exit(1);
            }
            cudaDeviceSynchronize();
            cuda_err = cudaGetLastError() ;
            if ( cudaSuccess != cuda_err ) {
                printf("cuda sync error = %s\n", cudaGetErrorString (cuda_err));
                exit(1) ;
            }
            GPU_Free_memory(&(d_input_string[stream_id]), &(d_r[stream_id]), &(d_hash_table[stream_id]), &(d_match_result[stream_id]), &(d_val_table[stream_id]), &(d_s0Table[stream_id]));
            cudaStreamDestroy(stream[stream_id]);
        }
    }

    clock_gettime( CLOCK_REALTIME, &test_e);
    tesTime = (test_e.tv_sec - test_b.tv_sec) * 1000.0;
    tesTime += (test_e.tv_nsec - test_b.tv_nsec) / 1000000.0;
    printf(" time for free memory  : %lf ms\n", tesTime);

    printf("/////////////////////////////////////////////\n");
    printf( "1.Time for  create PFAC : %lf seconds\n",  PFAC_duration);
    printf( "2.Time for  create Hashtable : %lf seconds\n",  Hashtable_duration);
    printf( "3.Time for  %d GPU malloc memory: %lf mseconds\n", GPU_N, (mallocGPU_duration)*1000 );
    printf( "4.Time for  %d GPU match progress: %lf mseconds\n", GPU_N, (mutiGPU_duration)*1000 );
    printf( "5.Total elapsed time: %lf mseconds\n",  (mallocGPU_duration + mutiGPU_duration)*1000);
//    printf( "kernel throughput  is %lf Gbps\n", double(input_size/mutiGPU_duration));
    printf("matching process finshed\n");
    printf("/////////////////////////////////////////////\n");

    cudaFreeHost(input_string);


    clock_gettime( CLOCK_REALTIME, &test_e1);
    tesTime1 = (test_e1.tv_sec - test_b1.tv_sec) * 1000.0;
    tesTime1 += (test_e1.tv_nsec - test_b1.tv_nsec) / 1000000.0;
    printf("time for all match process: %lf ms\n", tesTime1);

//    printf("cudaFreeHost(input_string); done\n");



    clock_gettime( CLOCK_REALTIME, &test_b);

    //merge match result from each thread
    int* match_result_aggreg = (int*)malloc(sizeof(int)*(size_t)input_size*(size_t)max_pat_len);
    memset(match_result_aggreg, 0xFF, sizeof(int)*(size_t)input_size*(size_t)max_pat_len);
    for (int GPUnum = 0; GPUnum < stream_N; GPUnum++) {
        for (i = 0; i < input_size; i++) {
            unsigned int k = (unsigned int)i * (unsigned int)max_pat_len;
            while(k < sizeof(int)*(size_t)input_size*(size_t)max_pat_len && match_result_aggreg[k] != -1) {
                k++;
            }
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
//        printf("match_result memory is freed, %d\n",GPUnum);
    }



    clock_gettime( CLOCK_REALTIME, &test_e);
    tesTime = (test_e.tv_sec - test_b.tv_sec) * 1000.0;
    tesTime += (test_e.tv_nsec - test_b.tv_nsec) / 1000000.0;
    printf("time for merge result: %lf ms\n", tesTime);


    //output match result file
    const char * output_file_name = "GPU_match_result.txt";
    FILE *fpout1 = fopen(output_file_name, "w");
    if (fpout1 == NULL) {
        perror("Open output file failed.\n");
        exit(1);
    }
    for (i = 0; i < input_size; i++) {
        for (j = 0; j < max_pat_len; j++){
            if (match_result_aggreg[(unsigned int)i*(unsigned int)max_pat_len+(unsigned int)j] != -1) {
                fprintf(fpout1, "At position %4d, match pattern %d\n", i, match_result_aggreg[(unsigned int)i*(unsigned int)max_pat_len+(unsigned int)j]);
            } else {
                break;
            }
        }
    }
    fclose(fpout1);
    return 0;
}
