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

//texture < int, 1, cudaReadModeElementType > tex_r;
//texture < int, 1, cudaReadModeElementType > tex_HT;

// First, look up s_s0Table to jump from initial state to next state.
// If the thread in still alive, then keep tracing HT (hash table)
// in texture until the thread be terminated (-1).
#define  SUBSEG_MATCH( j, match ) \
    pos = tid + j * BLOCK_SIZE ; \
    inputChar = s_in_c[pos]; \
    if (pos < input_size) {\
        state = s_s0Table[inputChar]; \
        yang123 = 0; \
        if (state >= 0) { \
            if (state < num_final_state) { \
                match[yang123] = state; \
                yang123++; \
            } \
            pos += 1; \
            while (1) { \
                if (pos >= bdy) break; \
                inputChar = s_in_c[pos]; \
                int key = (state << 8) + inputChar; \
                int row = key >> width_bit; \
                int col = key & ((1<<width_bit)-1); \
                int index = d_r[row] + col; \
                if(index < 0 || index >= HTSize) \
                    state = -1; \
                else { \
                      int hashValue = d_hash_table[index]; \
                      if ((hashValue) == row) \
                        state = d_val_table[index] ; \
                      else \
                        state = -1; \
                } \
                \
                if (state == -1) break; \
                if (state < num_final_state) { \
                  match[yang123] = state; \
                  yang123++; \
                } \
                if (yang123 > max_pat_len ){ \
                  printf("yang123 is bigger than maxlength in thread%d \n",tid); \
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
    int yang123;
    int inputChar;
    unsigned int *match[(PAGE_SIZE_C / BLOCK_SIZE)] = {0};   // registers to save match result
    for (int i = 0; i < (PAGE_SIZE_C / BLOCK_SIZE); i++) {
        match[i] = (unsigned int*)malloc(sizeof(unsigned int) * max_pat_len);
        for(int j = 0; j < max_pat_len; j++) {
            match[i][j] = - 1;
        }
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

    // save match result from registers to global memory
    start = gbid * PAGE_SIZE_C + tid;
    unsigned int d_match_size = (unsigned int)max_pat_len * (unsigned int)input_size;
    unsigned int thread_offset = (unsigned int)start * (unsigned int)max_pat_len;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        unsigned int i_offset = (unsigned int)i * (unsigned int)max_pat_len * (unsigned int)BLOCK_SIZE;
        for (int j = 0; j < max_pat_len; j++) {
            if(thread_offset + i_offset + (unsigned int)j < 0) printf("Overflow??\n");
            if(thread_offset + i_offset + (unsigned int)j < d_match_size) {
                d_match_result[thread_offset + i_offset + (unsigned int)j] = match[i][j];
//                printf("match result is %d\n", d_match_result[thread_offset + i_offset + (unsigned int)j]);
            }
            if(int(match[i][j])<-1) printf("???\n");
        }
        free(match[i]);
    }
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
    int MaxRow;
    MaxRow = (state_num*CHAR_SET) / width + 1;

    cudaMalloc((void **) d_input_string, num_blocks*PAGE_SIZE_C+EXTRA_SIZE_PER_TB*sizeof(int) );


    cudaMalloc((void **) d_r, MaxRow*sizeof(int) );


    cudaMalloc((void **) d_hash_table, HTSize*sizeof(int) );


    cudaMalloc((void **) d_val_table, HTSize*sizeof(int) );//add by qiao 0324


    cudaMalloc((void **) d_match_result, (size_t)max_pat_len*(size_t)input_size*sizeof(unsigned int));


    cudaMalloc((void **) d_s0Table, CHAR_SET*sizeof(int));

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda malloc memory error = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }


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
    printf("grid=(%d, %d), num_blocks=%d\n", dimGrid.x, dimGrid.y, num_blocks);
    printf("input_size = %d char\n", input_size );

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error0 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }

    // record time setting
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMemcpyAsync(d_input_string, input_string, input_size, cudaMemcpyHostToDevice, stream);
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error1 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    cudaMemcpyAsync(d_r, r, MaxRow*sizeof(int), cudaMemcpyHostToDevice, stream);
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error2 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    cudaMemcpyAsync(d_hash_table, HT, HTSize*sizeof(int), cudaMemcpyHostToDevice, stream);
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error3 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    cudaMemcpyAsync(d_s0Table, s0Table, CHAR_SET*sizeof(int), cudaMemcpyHostToDevice, stream);
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error4 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    cudaMemcpyAsync(d_val_table, val, HTSize*sizeof(int), cudaMemcpyHostToDevice, stream);//add by qiao 0324

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error5 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }

        // count bit of width (ex: if width is 256, width_bit is 8)
    int width_bit;
    for (width_bit = 0; (width >> width_bit)!=1; width_bit++);

//    cudaStreamSynchronize(stream);

    TraceTable_kernel <<< dimGrid, dimBlock, 0, stream >>> (d_match_result, (int *)d_input_string, input_size, HTSize,
        width_bit, final_state_num, MaxRow, num_blocks, boundary, d_s0Table, d_r, d_hash_table,
        d_val_table, max_pat_len);

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda kernel excute error = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }

//    cudaStreamSynchronize(stream);

    cudaMemcpyAsync(match_result, d_match_result, sizeof(int)*max_pat_len*input_size, cudaMemcpyDeviceToHost, stream);

    // record time setting
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("The elapsed time is %f ms\n", time);

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda memcpy error6 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }

    cudaStreamSynchronize(stream);

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda streamsync error = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
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
    printf("cudaFree(d_input_string); done\n");


    cudaFree(*d_r);

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda free memory error2 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    printf("cudaFree(d_r); done\n");


    //cudaUnbindTexture(tex_HT);
    cudaFree(*d_hash_table);
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda free memory error3 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    printf("cudaFree(d_hash_table); done\n");


    cudaFree(*d_val_table);//add by qiao0324
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda free memory error4 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    printf("cudaFree(d_val_table); done\n");


    cudaFree(*d_match_result);
    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda free memory error5 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    printf("cudaFree(d_match_result); done\n");


    cudaFree(*d_s0Table);
    printf("cudaFree(d_s0Table); done\n");

    cuda_err = cudaGetLastError() ;
    if ( cudaSuccess != cuda_err ) {
        printf("cuda free memory error6 = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }

    return 0;
};