#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE   512
#define PAGE_SIZE_I  1024   // size of a segment handled by a block (how many integers)
#define PAGE_SIZE_C  (PAGE_SIZE_I*sizeof(int))  // size of a segment handled by a block (how many bytes)
#define EXTRA_SIZE_PER_TB  128   // overlapd region size between segments (unit is integer)
#define CHAR_SET 256

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
            }
            if(int(match[i][j])<-1) printf("???\n");
        }
        free(match[i]);
    }
}

// First, look up s_s0Table to jump from initial state to next state.
// If the thread in still alive, then keep tracing HT (hash table)
// in texture until the thread be terminated (-1).
#define  SUBSEG_MATCH_FAST( j, match ) \
    pos = tid + j * BLOCK_SIZE ; \
    inputChar = s_in_c[pos]; \
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
            int index = d_r[state] + inputChar; \
            if (index >= HTSize) \
                state = -1; \
            else { \
                int hashValue = d_hash_table[index]; \
                if (hashValue == state) \
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
            pos += 1; \
        } \
    }

///****************************************************************************
//*   Function   : TraceTable_kernel_fast
//*   Description: This function trace PHF hash table to match input string.
//*                Because the width of key table is 256, some computation
//*                can be discarded.
//*   Parameters : d_match_result - Address to store match result
//*                d_in_i - Device (global) memory in int unit
//*                input_size - Size of input string
//*                HTSize - Size of hash table
//*                width_bit - Bits of key table width
//*                num_final_state - Total number of final states
//*                MaxRow - Total number of rows in key table
//*                num_blocks - Total number of blocks
//*                boundary - The last segment size
//*                d_s0Table - The row of initial state in PFAC table
//*   Returned   : No use
//****************************************************************************/
//__global__ void TraceTable_kernel_fast(unsigned int *d_match_result, int *d_in_i,
//                                       int input_size, int HTSize, int num_final_state, int MaxRow,
//                                       int num_blocks, int boundary, int *d_s0Table,
//                                       int* d_r, int* d_hash_table, int* d_val_table, int max_pat_len) {
//    int tid = threadIdx.x;
//    int gbid = blockIdx.y * gridDim.x + blockIdx.x;   // global block ID
//    int start = gbid * PAGE_SIZE_I + tid;
//    int pos;   // position to read input for the thread
//    int state;
//    int yang123;
//    int inputChar;
//    unsigned int match[(PAGE_SIZE_C / BLOCK_SIZE)][100] = {0};   // registers to save match result
//    unsigned char *s_in_c;   // shared memory in char unit
//    unsigned char *d_in_c;   // device (global) memory in char unit
//    int bdy;
//    __shared__ int s_in_i[PAGE_SIZE_I + EXTRA_SIZE_PER_TB];   // shared memory in int unit
//    __shared__ int s_s0Table[CHAR_SET];   // move the row of initial state in PFAC table to shared memory
//
//    if (gbid >= num_blocks) return;
//
//    s_in_c = (unsigned char *) s_in_i;
//    d_in_c = (unsigned char *) d_in_i;
//
//    pos = start;
//    // move data from global to shared memory
//    s_in_i[tid] = d_in_i[pos];
//    s_in_i[BLOCK_SIZE + tid] = d_in_i[BLOCK_SIZE + pos];
//    if (tid < EXTRA_SIZE_PER_TB) {
//        s_in_i[2 * BLOCK_SIZE + tid] = d_in_i[2 * BLOCK_SIZE + pos];
//    }
//    if (tid < CHAR_SET) {
//        s_s0Table[tid] = d_s0Table[tid];
//    }
//    __syncthreads();
//
//    if (gbid == num_blocks - 1)
//        bdy = boundary;
//    else
//        bdy = PAGE_SIZE_C + EXTRA_SIZE_PER_TB * sizeof(int);
//
//    // every thread handle (PAGE_SIZE_C/BLOCK_SIZE) position
//    SUBSEG_MATCH_FAST(0, match[0]);
//    SUBSEG_MATCH_FAST(1, match[1]);
//    SUBSEG_MATCH_FAST(2, match[2]);
//    SUBSEG_MATCH_FAST(3, match[3]);
//    SUBSEG_MATCH_FAST(4, match[4]);
//    SUBSEG_MATCH_FAST(5, match[5]);
//    SUBSEG_MATCH_FAST(6, match[6]);
//    SUBSEG_MATCH_FAST(7, match[7]);
//
//    // save match result from registers to global memory
//    start = gbid * PAGE_SIZE_C + tid;
//#pragma unroll
//    for (int i = 0; i < 8; i++) {
//        for (int j = 0; j < max_pat_len; j++) {
//            d_match_result[start*max_pat_len + i*max_pat_len*BLOCK_SIZE + j] = match[i][j];
//        }
//    }
//}

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
//int GPU_TraceTable(unsigned char *input_string, int input_size, int state_num,
//                   int final_state_num, unsigned int* match_result, int HTSize, int width,
//                   int *s0Table, int max_pat_len, int r[], int HT[], int val[])
//{
//
//
//    cudaGetDeviceCount(&GPU_S);
//
//
//
//
//
//
//
//        printf("1. H2D transfer time: %lf ms\n", transInTime);
//        printf("   H2D throughput: %lf GBps\n", (input_size+MaxRow*sizeof(int)+HTSize*sizeof(int)+CHAR_SET*sizeof(int))
//                                                /(transInTime*1000000));
//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after malloc memory7: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
//
//
//
//         //size_t free_mem, total_mem ;
//         //cudaError_t mem_info = cudaMemGetInfo( &free_mem, &total_mem);
//         //if ( cudaSuccess != mem_info ) {
//         //    printf("memory get info fails\n");
//         //    exit(1) ;
//         //}
//         //printf("total mem = %lf MB, free mem = %lf MB \n", total_mem/1024.0/1024.0 , free_mem/1024.0/1024.0 );
//
//        // set texture memory for hash table on device
//        // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc <int> ();  // another usage
//
//
//
//
//
//
//
////
////        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc (sizeof(int)*8, 0, 0, 0, cudaChannelFormatKindSigned);
////        cuda_err = cudaBindTexture(0, tex_r, d_r, channelDesc, MaxRow*sizeof(int));
////        if ( cudaSuccess != cuda_err ){
////            printf("cudaBindTexture on tex_r error\n");
////            exit(1) ;
////        }
////
////        cuda_err = cudaBindTexture(0, tex_HT, d_hash_table, channelDesc, HTSize*sizeof(int));
////        if ( cudaSuccess != cuda_err ){
////            printf("cudaBindTexture on tex_HT error\n");
////            exit(1) ;
////        }
//
//        // count bit of width (ex: if width is 256, width_bit is 8)
//
//
//
//        // release memory
//        cudaFree(d_input_string);
//        printf("cudaFree(d_input_string); done\n");
//
//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after free memory1: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
//        //cudaUnbindTexture(tex_r);
//        cudaFree(d_r);
//        printf("cudaFree(d_r); done\n");
//
//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after free memory2: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
//        //cudaUnbindTexture(tex_HT);
//        cudaFree(d_hash_table);
//        printf("cudaFree(d_hash_table); done\n");
//
//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after free memory3: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
//        cudaFree(d_val_table);//add by qiao0324
//        printf("cudaFree(d_val_table); done\n");
//
//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after free memory4: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
//        cudaFree(d_match_result);
//        printf("cudaFree(d_match_result); done\n");
//
//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after free memory5: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
//        cudaFree(d_s0Table);
//        printf("cudaFree(d_s0Table); done\n");
//
//
//        cuda_err = cudaGetLastError() ;
//        if ( cudaSuccess != cuda_err ) {
//            printf("after free memory6: error = %s\n", cudaGetErrorString (cuda_err));
//            exit(1) ;
//        }
//        // for(int testindex = 0; testindex < sizeof(short)*max_pat_len*input_size; testindex ++) {
//        //   if(match_result[testindex] < -1) printf("2Negative value %d at index %d\n", match_result[testindex], testindex);
//        // }
//
//        return 0 ;
//
//
//}
