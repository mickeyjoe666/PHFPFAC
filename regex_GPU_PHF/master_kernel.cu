#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE   512
#define PAGE_SIZE_I  1024   // size of a segment handled by a block (how many integers)
#define PAGE_SIZE_C  (PAGE_SIZE_I*sizeof(int))  // size of a segment handled by a block (how many bytes)
#define EXTRA_SIZE_PER_TB  128   // overlapd region size between segments (unit is integer)
#define initial_state (num_final_state+1)
#define CHAR_SET 256

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

texture < int, 1, cudaReadModeElementType > tex_r;
texture < int, 1, cudaReadModeElementType > tex_HT;
texture < int, 1, cudaReadModeElementType > tex_val;

// First, look up s_s0Table to jump from initial state to next state.
// If the thread in still alive, then keep tracing HT (hash table)
// in texture until the thread be terminated (-1).
#define  SUBSEG_MATCH( j, match ) \
    pos = tid + j * BLOCK_SIZE ; \
    inputChar = s_in_c[pos]; \
    if (pos < input_size) {\
        state = s_s0Table[inputChar]; \
        matchi = 0; \
        if (state >= 0) { \
            if (state < num_final_state) { \
                match[matchi] = state; \
                matchi++; \
            } \
            pos += 1; \
            while (1) { \
                if (pos >= bdy) break; \
                inputChar = s_in_c[pos]; \
                int key = (state << 8) + inputChar; \
                int row = key >> width_bit; \
                int col = key & ((1<<width_bit)-1); \
                int index = tex1Dfetch(tex_r, row) + col; \
                if(index < 0 || index >= HTSize) \
                    state = -1; \
                else { \
                      int hashValue = tex1Dfetch(tex_HT, index); \
                      if ((hashValue) == row) \
                        state = tex1Dfetch(tex_val, index); \
                      else \
                        state = -1; \
                } \
                \
                if (state == -1) break; \
                if (state < num_final_state) { \
                  match[matchi] = state; \
                  matchi++; \
                } \
                pos += 1; \
            } \
        }\
    } 
    

/****************************************************************************
*   Function   : TraceTable_kernel
*   Description: This function trace PHF hash table to match input string
*   Parameters : d_match_result - Address to store match result
*                d_in_i - Device (global) memory in int unit
*                input_size - Size of input string
*                HTSize - Size of hash table
*                width_bit - Bits of key table width
*                num_final_state - Total number of final states
*                MaxRow - Total number of rows in key table
*                num_blocks - Total number of blocks
*                boundary - The last segment size
*                d_s0Table - The row of initial state in PFAC table
*   Returned   : No use
****************************************************************************/
__global__ void TraceTable_kernel(unsigned int *d_match_result, int *d_in_i, int input_size,
                                  int HTSize, int width_bit, int num_final_state, int MaxRow,
                                  int num_blocks, int boundary, int *d_s0Table, int* d_r, int* d_hash_table,
                                  int* d_val_table, int max_pat_len) {
    int tid = threadIdx.x;
    int gbid = blockIdx.y * gridDim.x + blockIdx.x;   // global block ID
    int start = gbid * PAGE_SIZE_I + tid;
    int pos;   // position to read input for the thread
    int state;
    int matchi;
    int inputChar;

    unsigned int startTest = gbid * PAGE_SIZE_C + tid;
    unsigned int thread_offset = startTest * (unsigned int)max_pat_len;
    unsigned int *match[(PAGE_SIZE_C / BLOCK_SIZE)] = {0};   // registers to save match result
//    for (int i = 0; i < (PAGE_SIZE_C / BLOCK_SIZE); i++) {
//        match[i] = (unsigned int *) malloc(sizeof(unsigned int) * max_pat_len);
//        for (int j = 0; j < max_pat_len; j++) {
//            match[i][j] = -1;
//        }
//    }
    for (unsigned int i = 0; i < (PAGE_SIZE_C / BLOCK_SIZE); i++) {
        match[i] = &(d_match_result[thread_offset + i * (unsigned int)max_pat_len * (unsigned int)BLOCK_SIZE]);
    }
    unsigned char *s_in_c;   // shared memory in char unit
    unsigned char *d_in_c;   // device (global) memory in char unit
    int bdy;
    __shared__ int s_in_i[PAGE_SIZE_I + EXTRA_SIZE_PER_TB];   // shared memory in int unit
    __shared__ int s_s0Table[CHAR_SET];   // move the row of initial state in PFAC table to shared memory

    if (gbid >= num_blocks) return;

    s_in_c = (unsigned char *) s_in_i;
    d_in_c = (unsigned char *) d_in_i;

    pos = start;
    // move data from global to shared memory
    if(tid < PAGE_SIZE_I + EXTRA_SIZE_PER_TB)
      s_in_i[tid] = d_in_i[pos];
    if(BLOCK_SIZE + tid < PAGE_SIZE_I + EXTRA_SIZE_PER_TB)
      s_in_i[BLOCK_SIZE + tid] = d_in_i[BLOCK_SIZE + pos];
    if (tid < EXTRA_SIZE_PER_TB && 2* BLOCK_SIZE + tid < PAGE_SIZE_I + EXTRA_SIZE_PER_TB) {
        s_in_i[2 * BLOCK_SIZE + tid] = d_in_i[2 * BLOCK_SIZE + pos];
    }
    if (tid < CHAR_SET) {
        s_s0Table[tid] = d_s0Table[tid];
    }
    __syncthreads();

    if (gbid == num_blocks - 1)
        bdy = boundary;
    else
        bdy = PAGE_SIZE_C + EXTRA_SIZE_PER_TB * sizeof(int);

    //  every thread handle (PAGE_SIZE_C/BLOCK_SIZE = 8) position
    SUBSEG_MATCH(0, match[0]);
    SUBSEG_MATCH(1, match[1]);
    SUBSEG_MATCH(2, match[2]);
    SUBSEG_MATCH(3, match[3]);
    SUBSEG_MATCH(4, match[4]);
    SUBSEG_MATCH(5, match[5]);
    SUBSEG_MATCH(6, match[6]);
    SUBSEG_MATCH(7, match[7]);


//    SUBSEG_MATCH(0, test);
//    SUBSEG_MATCH(1, &(d_match_result[(unsigned int)start * (unsigned int)max_pat_len + (unsigned int)max_pat_len * (unsigned int)BLOCK_SIZE]));
//    SUBSEG_MATCH(2, &d_match_result[(unsigned int)start * (unsigned int)max_pat_len + (unsigned int) 2 * (unsigned int)max_pat_len * (unsigned int)BLOCK_SIZE]);
//    SUBSEG_MATCH(3, &d_match_result[(unsigned int)start * (unsigned int)max_pat_len + (unsigned int) 3 * (unsigned int)max_pat_len * (unsigned int)BLOCK_SIZE]);
//    SUBSEG_MATCH(4, &d_match_result[(unsigned int)start * (unsigned int)max_pat_len + (unsigned int) 4 * (unsigned int)max_pat_len * (unsigned int)BLOCK_SIZE]);
//    SUBSEG_MATCH(5, &d_match_result[(unsigned int)start * (unsigned int)max_pat_len + (unsigned int) 5 * (unsigned int)max_pat_len * (unsigned int)BLOCK_SIZE]);
//    SUBSEG_MATCH(6, &d_match_result[(unsigned int)start * (unsigned int)max_pat_len + (unsigned int) 6 * (unsigned int)max_pat_len * (unsigned int)BLOCK_SIZE]);
//    SUBSEG_MATCH(7, &d_match_result[(unsigned int)start * (unsigned int)max_pat_len + (unsigned int) 7 * (unsigned int)max_pat_len * (unsigned int)BLOCK_SIZE]);

    // save match result from registers to global memory
//    start = gbid * PAGE_SIZE_C + tid;
//    unsigned int d_match_size = (unsigned int)max_pat_len * (unsigned int)input_size;
//    unsigned int thread_offset = (unsigned int)start * (unsigned int)max_pat_len;
//    #pragma unroll
//    for (int i = 0; i < 8; i++) {
//        for (int j = 0; j < max_pat_len; j++) {
//            if(thread_offset + (unsigned int)j < d_match_size) {
//                d_match_result[thread_offset + (unsigned int)j] = match[i][j];
//            }
//        }
//        thread_offset += (unsigned int)max_pat_len * (unsigned int)BLOCK_SIZE;
//        free(match[i]);
//    }
}

// First, look up s_s0Table to jump from initial state to next state.
// If the thread in still alive, then keep tracing HT (hash table)
// in texture until the thread be terminated (-1).



int GPU_Malloc_Memory(thread_data dataset, unsigned char **d_input_string, int **d_r, int **d_hash_table, unsigned int **d_match_result, int **d_val_table, int **d_s0Table)
{
    unsigned char* input_string = dataset.input_string;
    int input_size = dataset.input_size;
    int state_num = dataset.state_num;
    int final_state_num = dataset.final_state_num;
    unsigned int*  match_result = dataset.match_result;
    int HTSize = dataset.HTSize;
    int width = dataset.width;
    int* s0Table = dataset.s0Table;
    int max_pat_len = dataset.max_pat_len;
    int* r = dataset.r;
    int* HT = dataset.HT;
    int* val = dataset.val;

    // num_blocks = number of blocks to cover input stream
    int num_blocks = (input_size + PAGE_SIZE_C-1) / PAGE_SIZE_C ;
    cudaError_t cuda_err;

    // allocate memory for input string and result
//    unsigned char *d_input_string;
//    int *d_r;
//    int *d_hash_table;
//    unsigned int *d_match_result;
//    int *d_val_table;//add by qiao 0324
//    int *d_s0Table;

    struct timespec test_b, test_e;
    double tesTime;

    clock_gettime( CLOCK_REALTIME, &test_b);

    int MaxRow;
    MaxRow = (state_num*CHAR_SET) / width + 1;

    cudaMalloc((void **) d_input_string, num_blocks*PAGE_SIZE_C+EXTRA_SIZE_PER_TB*sizeof(int) );


    cudaMalloc((void **) d_r, MaxRow*sizeof(int) );


    cudaMalloc((void **) d_hash_table, HTSize*sizeof(int) );


    cudaMalloc((void **) d_val_table, HTSize*sizeof(int) );//add by qiao 0324


    cudaMalloc((void **) d_match_result, (size_t)max_pat_len*(size_t)input_size*sizeof(unsigned int));
    cudaMemset((void*) *d_match_result, 0xFF, (size_t)max_pat_len*(size_t)input_size*sizeof(unsigned int));

    cudaMalloc((void **) d_s0Table, CHAR_SET*sizeof(int));

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda malloc memory error = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }

    clock_gettime( CLOCK_REALTIME, &test_e);
    tesTime = (test_e.tv_sec - test_b.tv_sec) * 1000.0;
    tesTime += (test_e.tv_nsec - test_b.tv_nsec) / 1000000.0;
    printf("time for malloc memory and memset: %lf ms\n", tesTime);



    return 0;



}



/****************************************************************************
*   Function   : GPU_TraceTable
*   Description: This function prepapre resources for GPU, and launch kernel
*                according to the width of key table
*   Parameters : input_string - Input string
*                input_size - Size of input string
*                state_num - Total number of statesPFAC_table.txt
*                final_state_num - Total number of final states
*                match_result - Address to store match result
*                HTSize - Size of hash table
*                width - The width of key table
*                s0Table - The row of initial state in PFAC table
*   Returned   : No use
****************************************************************************/


int GPU_TraceTable(thread_data dataset, cudaStream_t stream, unsigned char *d_input_string, int *d_r, int *d_hash_table, unsigned int *d_match_result, int *d_val_table, int *d_s0Table)
{
    unsigned char* input_string = dataset.input_string;
    int input_size = dataset.input_size;
    int state_num = dataset.state_num;
    int final_state_num = dataset.final_state_num;
    unsigned int*  match_result = dataset.match_result;
    int HTSize = dataset.HTSize;
    int width = dataset.width;
    int* s0Table = dataset.s0Table;
    int max_pat_len = dataset.max_pat_len;
    int* r = dataset.r;
    int* HT = dataset.HT;
    int* val = dataset.val;

    int MaxRow;
    MaxRow = (state_num*CHAR_SET) / width + 1;
    cudaError_t cuda_err;
    struct timespec transInTime_begin, transInTime_end;
    double transInTime;
    struct timespec transOutTime_begin, transOutTime_end, test_b, test_e;
    double transOutTime, tesTime;
    clock_gettime( CLOCK_REALTIME, &test_b);


    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc (sizeof(int)*8, 0, 0, 0, cudaChannelFormatKindSigned);

    cuda_err = cudaBindTexture(0, tex_r, d_r, channelDesc, MaxRow*sizeof(int));
    if ( cudaSuccess != cuda_err ){
        printf("cudaBindTexture on tex_r error\n");
        exit(1) ;
    }

    cuda_err = cudaBindTexture(0, tex_HT, d_hash_table, channelDesc, HTSize*sizeof(int));
    if ( cudaSuccess != cuda_err ){
        printf("cudaBindTexture on tex_HT error\n");
        exit(1) ;
    }

    cuda_err = cudaBindTexture(0, tex_val, d_val_table, channelDesc, HTSize*sizeof(int));
    if ( cudaSuccess != cuda_err ){
        printf("cudaBindTexture on tex_val error\n");
        exit(1) ;
    }

    clock_gettime( CLOCK_REALTIME, &test_e);
    tesTime = (test_e.tv_sec - test_b.tv_sec) * 1000.0;
    tesTime += (test_e.tv_nsec - test_b.tv_nsec) / 1000000.0;
    printf("texture time: %lf ms\n", tesTime);



    // set BLOCK_SIZE threads per block, set grid size automatically
    int dimBlock = BLOCK_SIZE ;

    // num_blocks = number of blocks to cover input stream
    int num_blocks = (input_size + PAGE_SIZE_C-1) / PAGE_SIZE_C ;

    // last segment may be less than a PAGE_SIZE_C
    int boundary = input_size - (num_blocks-1)*PAGE_SIZE_C;

    // num_blocks = p * 32768 + q
    int p = num_blocks / 32768 ;
    dim3  dimGrid ;

    dimGrid.x = num_blocks ;
    if ( p > 0 ){
        dimGrid.x = 32768 ;
        dimGrid.y = (num_blocks % 32768) ? (p + 1) : p ;
    }
//    printf("grid=(%d, %d), num_blocks=%d\n", dimGrid.x, dimGrid.y, num_blocks);
//    printf("input_size = %d char\n", input_size );

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error0 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }

    clock_gettime( CLOCK_REALTIME, &transInTime_begin);


    cudaMemcpy(d_input_string, input_string, input_size, cudaMemcpyHostToDevice);
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error1 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    cudaMemcpy(d_r, r, MaxRow*sizeof(int), cudaMemcpyHostToDevice);
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error2 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    cudaMemcpy(d_hash_table, HT, HTSize*sizeof(int), cudaMemcpyHostToDevice);
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error3 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    cudaMemcpy(d_s0Table, s0Table, CHAR_SET*sizeof(int), cudaMemcpyHostToDevice);
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error4 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    cudaMemcpy(d_val_table, val, HTSize*sizeof(int), cudaMemcpyHostToDevice);//add by qiao 0324

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error5 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }

    clock_gettime( CLOCK_REALTIME, &transInTime_end);
    transInTime = (transInTime_end.tv_sec - transInTime_begin.tv_sec) * 1000.0;
    transInTime += (transInTime_end.tv_nsec - transInTime_begin.tv_nsec) / 1000000.0;
    printf("1. H2D transfer time: %lf ms\n", transInTime);

        // count bit of width (ex: if width is 256, width_bit is 8)
    int width_bit;
    for (width_bit = 0; (width >> width_bit)!=1; width_bit++);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    TraceTable_kernel <<< dimGrid, dimBlock, 0>>> (d_match_result, (int *)d_input_string, input_size, HTSize,
        width_bit, final_state_num, MaxRow, num_blocks, boundary, d_s0Table, d_r, d_hash_table,
        d_val_table, max_pat_len);

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda kernel excute error = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }

    // record time setting
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("2. MASTER: The elapsed time is %f ms\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

//    cudaStreamSynchronize(stream);

    clock_gettime( CLOCK_REALTIME, &transOutTime_begin);
    cudaMemcpy(match_result, d_match_result, sizeof(int)*max_pat_len*input_size, cudaMemcpyDeviceToHost);
    clock_gettime( CLOCK_REALTIME, &transOutTime_end);
    transOutTime = (transOutTime_end.tv_sec - transOutTime_begin.tv_sec) * 1000.0;
    transOutTime += (transOutTime_end.tv_nsec - transOutTime_begin.tv_nsec) / 1000000.0;
    printf("3. D2H transfer time: %lf ms\n", transOutTime);
    printf("4. Total elapsed time: %lf ms\n", transInTime+transOutTime+time);


    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error6 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda streamsync error = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    clock_gettime( CLOCK_REALTIME, &test_e);
    tesTime = (test_e.tv_sec - test_b.tv_sec) * 1000.0;
    tesTime += (test_e.tv_nsec - test_b.tv_nsec) / 1000000.0;
    printf("full time: %lf ms\n", tesTime);

    return 0 ;


}

int GPU_Free_memory(unsigned char **d_input_string, int **d_r, int **d_hash_table, unsigned int **d_match_result, int **d_val_table, int **d_s0Table)
{
    cudaError_t cuda_err;
    // release memory
    cudaFree(*d_input_string);

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda free memory error1 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
//    printf("cudaFree(d_input_string); done\n");


    cudaFree(*d_r);

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda free memory error2 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
//    printf("cudaFree(d_r); done\n");


    //cudaUnbindTexture(tex_HT);
    cudaFree(*d_hash_table);
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda free memory error3 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
//    printf("cudaFree(d_hash_table); done\n");


    cudaFree(*d_val_table);//add by qiao0324
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda free memory error4 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
//    printf("cudaFree(d_val_table); done\n");


    cudaFree(*d_match_result);
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda free memory error5 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
//    printf("cudaFree(d_match_result); done\n");


    cudaFree(*d_s0Table);
//    printf("cudaFree(d_s0Table); done\n");

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda free memory error6 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }

    cudaUnbindTexture(tex_r);
    cudaUnbindTexture(tex_val);
    cudaUnbindTexture(tex_HT);


    return 0;
};