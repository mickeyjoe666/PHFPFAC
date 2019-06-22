cmake_minimum_required(VERSION 3.8)
project(PHFPFAC)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        regex_GPU_PHF/CreateTable/charset_table.c
        regex_GPU_PHF/CreateTable/charset_table_reorder.c
        regex_GPU_PHF/CreateTable/create_PFAC_table_reorder.c
        regex_GPU_PHF/CreateTable/create_table_reorder.c
        regex_GPU_PHF/CreateTable/ctdef.h
        regex_GPU_PHF/PHF/phf.c
        regex_GPU_PHF/main.cc
        regex_GPU_PHF/master_kernel.cu)

add_executable(PHFPFAC ${SOURCE_FILES})