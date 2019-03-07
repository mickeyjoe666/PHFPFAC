#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "create_table_reorder.c"

int create_PFAC_table_reorder(char *patternfilename, int *state_num, int *final_state_num, int type, int *max_pat_len_arr, int *max_pat_len, int*** PFACs, int** patternIdMaps) {

    if (type >= 0 && type <= 1) {
        create_table_reorder(patternfilename, state_num, final_state_num, type, max_pat_len_arr, max_pat_len, PFACs, patternIdMaps);
    }
    else {
        printf("Only type 0 and 1 are supported.\n");
        exit (1);
    }
    
    return 0;
}
