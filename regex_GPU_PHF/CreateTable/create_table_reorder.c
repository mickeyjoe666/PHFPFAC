#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ctdef.h"

//pattern_s all_pattern[MAX_STATE];
//int pattern_num;
int GPU_N =2;

/****************************************************************************
*   Function   : comp_pat
*   Description: This function compare 2 pattern structure for qsort()
*   Parameters : a - 1st pattern
*                b - 2nd pattern
*   Returned   : Comparison result
****************************************************************************/
int comp_pat(const void *a, const void*b) {
    pattern_s *pat1 = (pattern_s *)a;
    pattern_s *pat2 = (pattern_s *)b;
    int str_len1;
    int str_len2;
    int min_len;
    int result;
    
    str_len1 = pat1->pattern_len;
    str_len2 = pat2->pattern_len;
    min_len = (str_len1 < str_len2) ? str_len1 : str_len2;
    
    result = memcmp(pat1->pat , pat2->pat, min_len);

    if (result == 0) {
        if (str_len1 > str_len2)
            return 1;
        else if (str_len1 < str_len2)
            return -1;
        else
            return 0;
    }
    
    return result;
}

/****************************************************************************
*   Function   : read_pattern
*   Description: This function read patterns from file to all_pattern[]
*   Parameters : patternfilename - Pattern file name string
*   Returned   : No use
****************************************************************************/
pattern_s* read_pattern(char *patternfilename, int* pattern_num, pattern_s all_pattern[]) {
    int ch;
    char str[1024];  // pattern length must less than 1024 in PFAC algo
    int str_len;
    FILE *fpin;
    
    *pattern_num = 0;
    
    // open input file
    fpin = fopen(patternfilename, "rb");
    if (fpin == NULL) {
        perror("Open input file failed.");
        exit(1);
    }
    
    while (1) {
        // read a pattern
        str_len = 0;
        while (1) {
            ch = fgetc(fpin);
            str[str_len++] = ch;
            
            if (str_len >= 1024) {
                printf("Pattern %d length over 1024.\n", *pattern_num+1);
                exit(1);
            }
            
            if (ch == '\n') {
                str_len -= 1;
                *pattern_num += 1;
                break;
            }
        }
        
        // put the read pattern to all_pattern[]
        all_pattern[*pattern_num].pattern_id = *pattern_num;
        all_pattern[*pattern_num].pattern_len = str_len;
        all_pattern[*pattern_num].pat = (char *)malloc( str_len*sizeof(char) );
        memcpy(all_pattern[*pattern_num].pat, str, str_len*sizeof(char));
        
        // check end-of-file
        ch = fgetc(fpin);
        if ( feof(fpin) ) {
            break;
        }
        else {
            ungetc(ch, fpin);
        }
    }
    
    // sort the patterns for correctness of creating table
    qsort(&all_pattern[1], *pattern_num, sizeof(pattern_s), comp_pat); 
    
    fclose(fpin);
}

/****************************************************************************
*   Function   : read_pattern_ext
*   Description: This function read patterns from file to all_pattern[]
                 using fgetc_ext() instead of fgetc()
*   Parameters : patternfilename - Pattern file name string
*   Returned   : No use
****************************************************************************/
void read_pattern_ext(char *patternfilename, int* pattern_num, pattern_s all_pattern[]) {
    int ch;
    char str[1024];  // pattern length must less than 1024 in PFAC algo
    int str_len;
    FILE *fpin;
    
    *pattern_num = 0;
    
    // open input file
    fpin = fopen(patternfilename, "rb");
    if (fpin == NULL) {
        perror("Open input file failed.");
        exit(1);
    }
    
    while (1) {
        // read a pattern
        str_len = 0;
        while (1) {
            ch = fgetc_ext(fpin);
            str[str_len++] = ch;
            
            if (str_len >= 1024) {
                printf("Pattern %d length over 1024.\n", *pattern_num+1);
                exit(1);
            }
            
            if (ch == EOL) {
                str_len -= 1;
                *pattern_num += 1;
                break;
            }
        }
        
        // put the read pattern to all_pattern[]
        all_pattern[*pattern_num].pattern_id = *pattern_num;
        all_pattern[*pattern_num].pattern_len = str_len;
        all_pattern[*pattern_num].pat = (char *)malloc( str_len*sizeof(char) );
        memcpy(all_pattern[*pattern_num].pat, str, str_len*sizeof(char));
        
        // check end-of-file
        ch = fgetc(fpin);
        if ( feof(fpin) ) {
            break;
        }
        else {
            ungetc(ch, fpin);
        }
    }
    
    // sort the patterns for correctness of creating table
    qsort(&all_pattern[1], *pattern_num, sizeof(pattern_s), comp_pat); 
    
    fclose(fpin);
}

/****************************************************************************
*   Function   : create_table_reorder
*   Description: create transition table from all_pattern[]
*   Parameters : patternfilename - Pattern file name string
*                state_num - Address of variable to store total number of
*                            state
*                final_state_num - Address of variable to store total number
*                                  of final state
*                ext - Extension mode selection. 0 for normal ASCII and
*                      1 for reading escape character
*                max_pat_length - Address of variable to store maximum
*                      pattern length
*   Returned   : No use (total number of state)
****************************************************************************/
int create_table_reorder(char *patternfilename, int *state_num, int *final_state_num, int ext, int* max_pat_length_arr, int* max_pat_len, int *** PFACs, int** patternIdMaps) {
    int i, j, x;
    int ch;
    int state;          // to traverse transition table
    int state_count;    // counter for creating new state
    int initial_state, pattern_num;
    //pattern_s all_pattern[MAX_STATE];
    pattern_s* all_pattern = (pattern_s*)malloc(MAX_STATE*sizeof(pattern_s));    

    // select normal mode or extension mode
    if (ext == 0)
        read_pattern(patternfilename, &pattern_num, all_pattern);
    else
        read_pattern_ext(patternfilename, &pattern_num, all_pattern);
 
    //cudaGetDeviceCount(&GPU_N);

    //the number of patterns to feed to GPUs 0 to GPU_N-2
    int k = pattern_num/GPU_N;
    //the number of patterns to ffed to GPU GPU_N-1. (GPU_N-2)*k + l = pattern_num.
    int l = k + pattern_num%GPU_N;
    //Allocate and initialise memory for the PFACs
    for (x =0 ; x < GPU_N ; x++){
        PFACs[x] = (int**)malloc(MAX_STATE*sizeof(int*));
        for (i = 0; i < MAX_STATE; i++) {
            PFACs[x][i] =(int*) malloc(CHAR_SET*sizeof(int));
            for (j = 0; j < CHAR_SET; j++) {
                (PFACs[x])[i][j] = -1;
            }
        }
    }


    //Array of array of patterns. Each divided_pattenrs[i] corresponds to the patterns of each GPU_i
    pattern_s** divided_patterns = divide_patterns(all_pattern, pattern_num);

    printf("k = %d, l= %d\n", k,l);

    for (i = 0;i< GPU_N-1; i++){
         patternIdMaps[i] = (int*)malloc(k*sizeof(int));
         patternsToPFAC(divided_patterns[i], k, PFACs[i], &(max_pat_length_arr[i]), &(state_num[i]), patternIdMaps[i]);
         if (max_pat_length_arr[i] > *max_pat_len) *max_pat_len = max_pat_length_arr[i];
         final_state_num[i] = k;
    }


    


    patternIdMaps[i] = (int*)malloc(l*sizeof(int));
    patternsToPFAC(divided_patterns[i], l, PFACs[i], &(max_pat_length_arr[i]), &(state_num[i]), patternIdMaps[i]);
    if (max_pat_length_arr[i] > *max_pat_len) *max_pat_len = max_pat_length_arr[i];
    final_state_num[i] = l;
    
}

pattern_s** divide_patterns(pattern_s all_pattern[], int pattern_num) {
    pattern_s** result;
    result = (pattern_s**)malloc(GPU_N * sizeof(pattern_s*));
    int i, j, k, l;
    k = pattern_num/GPU_N;
    l = k + pattern_num%GPU_N;

    for (i = 0; i <GPU_N-1; i++){
        result[i] = (pattern_s*)malloc(k* sizeof(pattern_s));
        for (j = 0; j <k; j++){
            //all_pattern is indexed from 1.
            result[i][j] = all_pattern[i*k+j+1];
        }

    }

    result[i] = (pattern_s*)malloc(l*sizeof(pattern_s));
    for(j=0; j< l; j++){
         result[i][j] = all_pattern[i*k+j+1];
    }
    return result;
}


void patternsToPFAC(pattern_s patterns[], int pattern_num, int** PFAC, int* max_pat_length, int *state_num, int patternIdMap[]){

    int state_count;
    int state;
    int initial_state;
    int ch;
    pattern_s cur_pat;
    int j;

    // final states are state[1] ~ state[n], n is number of pattern
    initial_state = pattern_num + 1;
    // state start from initial state
    state = initial_state;
    // create new state from (initial_state+1)
    state_count = initial_state + 1;

    for (int i = 0; i < pattern_num; i++) {
        // load current pattern
        cur_pat = patterns[i];
        patternIdMap[i] = cur_pat.pattern_id;
        if (cur_pat.pattern_len > *max_pat_length ) {
           *max_pat_length = cur_pat.pattern_len;
        } 
        //create transition according to pattern
        for (j = 0; j < cur_pat.pattern_len-1; j++) {
            ch = (unsigned char)cur_pat.pat[j];
            if (PFAC[state][ch] == -1) {
                PFAC[state][ch] = state_count;
                state = state_count;
                state_count += 1;
            }
            else {
                state = PFAC[state][ch];
            }
        }
        // the ending char will create a transition to corresponding final state
        ch = (unsigned char)cur_pat.pat[j];
        PFAC[state][ch] = i;
        // initialize state to load next pattern
        state = initial_state;
        
        // check state overflow
        if (state_count > MAX_STATE) {
            fprintf(stderr, "State number overflow, %d\n", state_count);
            exit(1);
        }
    }
    *state_num = state_count;
}
