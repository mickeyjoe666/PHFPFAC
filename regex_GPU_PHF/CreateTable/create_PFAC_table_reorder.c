#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "create_table_reorder.c"

int create_PFAC_table_reorder(char *patternfilename, int *state_num, int *final_state_num, int streamnum, int *max_pat_len_arr, int *max_pat_len, int*** PFACs, int** patternIdMaps, int GPU_use ) {

        create_table_reorder(patternfilename, state_num, final_state_num, streamnum, max_pat_len_arr, max_pat_len, PFACs, patternIdMaps, GPU_use);
    
    return 0;
}
