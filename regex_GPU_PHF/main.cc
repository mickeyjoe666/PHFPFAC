#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>
#include "CreateTable/create_PFAC_table_reorder.c"
#include "PHF/phf.c"

//int num_output[MAX_STATE];            // num of matched pattern for each state
//int *outputs[MAX_STATE];              // list of matched pattern for each state

//int r[ROW_MAX];          // r[R]=amount row Keys[R][] was shifted
//int HT[HASHTABLE_MAX];   // the shifted rows of Keys[][] collapse into HT[]
//int val[HASHTABLE_MAX];  // store next state corresponding to hash key, not used in this version

//int GPU_TraceTable(unsigned char *input_string, int input_size, int state_num,
//                   int final_state_num, unsigned int* match_result, int HTSize, int width,
//                   int *s0Table, int max_pat_len, int r[], int HT[], int val[]);

/****************************************************************************
*   Function   : main
*   Description: Main function
*   Parameters : Command line arguments
*   Returned   : Program end success(0) or fail(1)
****************************************************************************/
int main(int argc, char *argv[]) {
    //number of GPUs on the machine
    cudaGetDeviceCount(&GPU_S);
    int GPU_N = 3*GPU_S ;

    //Array contaning the number of states in the automaton of each GPU
    int* state_num = (int*)malloc(GPU_N*sizeof(int));
    //ArrayN contaning the number of final states in the automaton of each GPU
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
    int stream_num = 3;

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
    for(int GPUnum = 0; GPUnum < GPU_S; GPUnum++){
        if ( cudaSetDevice(GPUnum) != cudaSuccess ) {
            fprintf(stderr, "Set CUDA device %d error\n", GPUnum);
            exit(1);
        }

        // allocate host memory: match result
        //status = cudaMallocHost((void **) &(match_result[GPUnum]), sizeof(unsigned int)*input_size*max_pat_len_arr[GPUnum]);
        status = cudaHostAlloc((void **) &(match_result[GPUnum]), sizeof(unsigned int)*input_size*max_pat_len_arr[GPUnum], cudaHostAllocPortable);
        if (cudaSuccess != status) {
            fprintf(stderr, "cudaMallocHost match_result error: %s\n", cudaGetErrorString(status));
            exit(1);
        }
    }




//    for(int GPUnum = 0; GPUnum < GPU_N; GPUnum++){
//        GPU_TraceTable(input_string, input_size, state_num[GPUnum], final_state_num[GPUnum],
//                       match_result[GPUnum], HTSize[GPUnum], width, PFACs[GPUnum][(final_state_num[GPUnum]+1)],
//                       max_pat_len_arr[GPUnum], r[GPUnum], HT[GPUnum], val[GPUnum]);
//    }




    for(int i =0; i < GPU_S; i++) {
        if (cudaSetDevice(i) != cudaSuccess) {
            fprintf(stderr, "Set CUDA device %d error\n", GPUnum);
            exit(1);
        }


        cudaError_t cuda_err;
        struct timespec transInTime_begin, transInTime_end;
        double transInTime;
        struct timespec transOutTime_begin, transOutTime_end;
        double transOutTime;
        size_t free_mem;
        size_t total_mem;

        cuda_err = cudaGetLastError();
        if (cudaSuccess != cuda_err) {
            printf("before malloc memory: error = %s\n", cudaGetErrorString(cuda_err));
            exit(1);
        }

        // set BLOCK_SIZE threads per block, set grid size automatically
        int dimBlock = BLOCK_SIZE;

        // num_blocks = number of blocks to cover input stream
        int num_blocks = (input_size + PAGE_SIZE_C - 1) / PAGE_SIZE_C;

        // last segment may be less than a PAGE_SIZE_C
        int boundary = input_size - (num_blocks - 1) * PAGE_SIZE_C;

        // num_blocks = p * 32768 + q
        int p = num_blocks / 32768;
        dim3 dimGrid;

        dimGrid.x = num_blocks;
        if (p > 0) {
            dimGrid.x = 32768;
            dimGrid.y = (num_blocks % 32768) ? (p + 1) : p;
        }
        printf("grid=(%d, %d), num_blocks=%d\n", dimGrid.x, dimGrid.y, num_blocks);
        printf("input_size = %d char\n", input_size);


        // allocate memory for input string and result
        unsigned char *d_input_string;
        int *d_r_1;
        int *d_r_2;
        int *d_r_3;

        int *d_hash_table_1;
        int *d_hash_table_2;
        int *d_hash_table_3;

        unsigned int *d_match_result_1;
        unsigned int *d_match_result_2;
        unsigned int *d_match_result_3;

        int *d_val_table_1;//add by qiao 0324
        int *d_val_table_2;//add by qiao 0324
        int *d_val_table_3;//add by qiao 0324

        int *d_s0Table_1;
        int *d_s0Table_2;
        int *d_s0Table_3;
        int MaxRow;

        int GPU_num = stream_num*i+1;

        //create stream
        cudaStreamcudaStream_t stream1, stream2, stream3;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);

        MaxRow_1 = (state_num[GPU_num] * CHAR_SET) / width + 1;
        MaxRow_2 = (state_num[GPU_num] * CHAR_SET) / width + 1;
        MaxRow_3 = (state_num[GPU_num] * CHAR_SET) / width + 1;

        cudaMalloc((void **) &d_input_string, num_blocks * PAGE_SIZE_C + EXTRA_SIZE_PER_TB * sizeof(int));


        //malloc memory on device

//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after malloc memory1: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
        cudaMalloc((void **) &d_r_1, MaxRow_1 * sizeof(int));
        cudaMalloc((void **) &d_r_2, MaxRow_2 * sizeof(int));
        cudaMalloc((void **) &d_r_3, MaxRow_3 * sizeof(int));

//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after malloc memory2: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
        cudaMalloc((void **) &d_hash_table_1, HTSize * sizeof(int));
        cudaMalloc((void **) &d_hash_table_2, HTSize * sizeof(int));
        cudaMalloc((void **) &d_hash_table_3, HTSize * sizeof(int));

//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after malloc memory3: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
        cudaMalloc((void **) &d_val_table_1, HTSize * sizeof(int));//add by qiao 0324
        cudaMalloc((void **) &d_val_table_2, HTSize * sizeof(int));//add by qiao 0324
        cudaMalloc((void **) &d_val_table_3, HTSize * sizeof(int));//add by qiao 0324

//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after malloc memory4: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }

        cudaError_t mem_info1 = cudaMemGetInfo(&free_mem, &total_mem);
        if (cudaSuccess != mem_info1) {
            printf("memory get info fails\n");
            exit(1);
        }
//        printf("total mem = %lf MB, free mem before malloc d_match_result = %lf MB \n", total_mem/1024.0/1024.0 , free_mem/1024.0/1024.0 );
//        printf("Trying to allocate %lu bits of memory\n", (size_t)max_pat_len*(size_t)input_size*sizeof(short));
//        printf("Test %u\n", (unsigned int)max_pat_len*(unsigned int)input_size);
//        printf("max_pat_len: %d, input_size: %d, sizeof(unsigned int): %lu", max_pat_len, input_size, sizeof(unsigned int));


        cudaMalloc((void **) &d_match_result_1,
                   (size_t) max_pat_len_arr[GPU_num] * (size_t) input_size * sizeof(unsigned int));
        cudaMalloc((void **) &d_match_result_2,
                   (size_t) max_pat_len_arr[GPU_num+1] * (size_t) input_size * sizeof(unsigned int));
        cudaMalloc((void **) &d_match_result_3,
                   (size_t) max_pat_len_arr[GPU_num+2] * (size_t) input_size * sizeof(unsigned int));


        cudaError_t mem_info2 = cudaMemGetInfo(&free_mem, &total_mem);
        if (cudaSuccess != mem_info2) {
            printf("memory get info fails\n");
            exit(1);
        }
//        printf("total mem = %lf MB, free mem after malloc d_match_result = %lf MB \n", total_mem/1024.0/1024.0 , free_mem/1024.0/1024.0 );

//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after malloc memory5: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }

        cudaMalloc((void **) &d_s0Table_1, CHAR_SET * sizeof(int));
        cudaMalloc((void **) &d_s0Table_2, CHAR_SET * sizeof(int));
        cudaMalloc((void **) &d_s0Table_3, CHAR_SET * sizeof(int));

//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after malloc memory6: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }

    }



    for (int i = 0; i < GPU_S; i++) {
        if (cudaSetDevice(i) != cudaSuccess) {
            fprintf(stderr, "Set CUDA device %d error\n", GPUnum);
            exit(1);
        }
        int GPU_num = stream_num * i + 1;

        clock_gettime(CLOCK_REALTIME, &transInTime_begin);
        // copy input string from host to device
        cudaMemcpy(d_input_string, input_string, input_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_string, input_string, input_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_string, input_string, input_size, cudaMemcpyHostToDevice);

        cudaMemcpyAsync(d_r_1, r[GPU_num], MaxRow_1 * sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_r_2, r[GPU_num + 1], MaxRow_2 * sizeof(int), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_r_3, r[GPU_num + 2], MaxRow_3 * sizeof(int), cudaMemcpyHostToDevice, stream3);

        cudaMemcpyAsync(d_hash_table_1, HT[GPU_num], HTSize[GPUnum] * sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_hash_table_2, HT[GPU_num + 1], HTSize[GPUnum + 1] * sizeof(int), cudaMemcpyHostToDevice,
                        stream2);
        cudaMemcpyAsync(d_hash_table_3, HT[GPU_num + 2], HTSize[GPUnum + 2] * sizeof(int), cudaMemcpyHostToDevice,
                        stream3);

        cudaMemcpyAsync(d_s0Table_1, PFACs[GPUnum][(final_state_num[GPUnum] + 1)], CHAR_SET * sizeof(int),
                        cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_s0Table_2, PFACs[GPUnum + 1][(final_state_num[GPUnum + 1] + 1)], CHAR_SET * sizeof(int),
                        cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_s0Table_3, PFACs[GPUnum + 2][(final_state_num[GPUnum + 2] + 1)], CHAR_SET * sizeof(int),
                        cudaMemcpyHostToDevice, stream3);

        cudaMemcpyAsync(d_val_table_1, val[GPU_num], HTSize[GPUnum] * sizeof(int), cudaMemcpyHostToDevice,
                        stream1);//add by qiao 0324
        cudaMemcpyAsync(d_val_table_2, val[GPU_num + 1], HTSize[GPUnum + 1] * sizeof(int), cudaMemcpyHostToDevice,
                        stream2);//add by qiao 0324
        cudaMemcpyAsync(d_val_table_3, val[GPU_num + 2], HTSize[GPUnum + 2] * sizeof(int), cudaMemcpyHostToDevice,
                        stream3);//add by qiao 0324



        clock_gettime(CLOCK_REALTIME, &transInTime_end);
        transInTime = (transInTime_end.tv_sec - transInTime_begin.tv_sec) * 1000.0;
        transInTime += (transInTime_end.tv_nsec - transInTime_begin.tv_nsec) / 1000000.0;

        printf("1. H2D transfer time: %lf ms\n", transInTime);
        printf("   H2D throughput: %lf GBps\n",
               (input_size + MaxRow * sizeof(int) + HTSize * sizeof(int) + CHAR_SET * sizeof(int))
               / (transInTime * 1000000));

        int width_bit;
        for (width_bit = 0; (width >> width_bit) != 1; width_bit++);

        // check error before kernel launch
        cuda_err = cudaGetLastError();
        if (cudaSuccess != cuda_err) {
            printf("before kernel call: error = %s\n", cudaGetErrorString(cuda_err));
            exit(1);
        }

        // record time setting
        cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);//add by qiao 20190402


        //run 3 stream on the same device
        TraceTable_kernel <<< dimGrid, dimBlock, stream1 >>>
                                                  (d_match_result_1, (int *) d_input_string, input_size, HTSize[GPUnum],
                                                          width_bit, final_state_num[GPUnum], MaxRow_1, num_blocks, boundary, d_s0Table_1, d_r_1, d_hash_table_1, d_val_table_1, max_pat_len_arr[GPUnum]);

        TraceTable_kernel <<< dimGrid, dimBlock, stream2 >>>
                                                  (d_match_result_2, (int *) d_input_string, input_size, HTSize[GPUnum +
                                                                                                                1],
                                                          width_bit, final_state_num[GPUnum +
                                                                                     1], MaxRow_2, num_blocks, boundary, d_s0Table_2, d_r_2, d_hash_table_2, d_val_table_2, max_pat_len_arr[
                                                          GPUnum + 1]);

        TraceTable_kernel <<< dimGrid, dimBlock, stream3 >>>
                                                  (d_match_result_3, (int *) d_input_string, input_size, HTSize[GPUnum +
                                                                                                                2],
                                                          width_bit, final_state_num[GPUnum +
                                                                                     2], MaxRow_3, num_blocks, boundary, d_s0Table_3, d_r_3, d_hash_table_3, d_val_table_3, max_pat_len_arr[
                                                          GPUnum + 2]);





        // check error after kernel launch
        cuda_err = cudaGetLastError();
        if (cudaSuccess != cuda_err) {
            printf("after kernel call: error = %s\n", cudaGetErrorString(cuda_err));
            exit(1);
        }

        // record time setting
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("2. MASTER: The elapsed time is %f ms\n", time);
        printf("   MASTER: The throughput is %f Gbps\n", (float) (input_size) / (time * 1000000) * 8);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        clock_gettime(CLOCK_REALTIME, &transOutTime_begin);

        cuda_err = cudaGetLastError();
        if (cudaSuccess != cuda_err) {
            printf("cudSucess is %d\n, cuda_err is %d\n ", cudaSuccess, cuda_err);
            printf("before malloc match result memory: error = %s\n", cudaGetErrorString(cuda_err));
            exit(1);
        }

        cudaMemcpyAsync(match_result[GPU_num], d_match_result_1,
                        sizeof(unsigned int) * max_pat_len_arr[GPUnum] * input_size, cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(match_result[GPU_num + 1], d_match_result_2,
                        sizeof(unsigned int) * max_pat_len_arr[GPUnum + 1] * input_size, cudaMemcpyDeviceToHost,
                        stream2);
        cudaMemcpyAsync(match_result[GPU_num + 2], d_match_result_3,
                        sizeof(unsigned int) * max_pat_len_arr[GPUnum + 2] * input_size, cudaMemcpyDeviceToHost,
                        stream3);

        //wait all stream done
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        cudaStreamSynchronize(stream3);



//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after malloc memory8: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
        // for(int testindex = 0; testindex < sizeof(short)*max_pat_len*input_size; testindex ++) {
        //   if(match_result[testindex] < -1) printf("Negative value %d at index %d\n", match_result[testindex], testindex);
        // }
        clock_gettime(CLOCK_REALTIME, &transOutTime_end);
        transOutTime = (transOutTime_end.tv_sec - transOutTime_begin.tv_sec) * 1000.0;
        transOutTime += (transOutTime_end.tv_nsec - transOutTime_begin.tv_nsec) / 1000000.0;
        printf("3. D2H transfer time: %lf ms\n", transOutTime);
        printf("   D2H throughput: %lf GBps\n", (input_size * sizeof(unsigned int)) / (transOutTime * 1000000));

        printf("4. Total elapsed time: %lf ms\n", transInTime + transOutTime + time);
        printf("   Total throughput: %lf Gbps\n",
               (double) input_size / ((transInTime + transOutTime + time) * 1000000) * 8);
        printf("///////////////////////////////////////////////////////\n");

        cudaFree(d_input_string);
//        printf("cudaFree(d_input_string); done\n");

//        cuda_err = cudaGetLastError();
//        if ( cudaSuccess != cuda_err ) {
//            printf("after free memory1: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
        //cudaUnbindTexture(tex_r);
        cudaFree(d_r_1);
        cudaFree(d_r_2);
        cudaFree(d_r_3);
//        printf("cudaFree(d_r); done\n");

//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after free memory2: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
        //cudaUnbindTexture(tex_HT);
        cudaFree(d_hash_table_1);
        cudaFree(d_hash_table_2);
        cudaFree(d_hash_table_3);
//        printf("cudaFree(d_hash_table); done\n");

//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after free memory3: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
        cudaFree(d_val_table_1);//add by qiao0324
        cudaFree(d_val_table_2);//add by qiao0324
        cudaFree(d_val_table_3);//add by qiao0324
//        printf("cudaFree(d_val_table); done\n");

//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after free memory4: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
        cudaFree(d_match_result_1);
        cudaFree(d_match_result_2);
        cudaFree(d_match_result_3);
//        printf("cudaFree(d_match_result); done\n");

//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after free memory5: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
        cudaFree(d_s0Table_1);
        cudaFree(d_s0Table_2);
        cudaFree(d_s0Table_3);
//        printf("cudaFree(d_s0Table); done\n");


//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after free memory6: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//
//
//        }


    }

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

    //free all stream on device
    for(int i =0; i < GPU_S; i++){

        if (cudaSetDevice(i) != cudaSuccess) {
            fprintf(stderr, "Set CUDA device %d error\n", GPUnum);
            exit(1);
        }

        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream3);

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
