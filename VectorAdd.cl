/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 // OpenCL Kernel Function for element by element vector addition
__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
	
		     // Get the index of the current element to be processed 
		     int i = get_global_id(0);
	
		     // Do the operation 
			// C[i] = (int)((A[i] + B[i])*35.222/59.345*0.123);
			 C[i] = (A[i] + B[i]);
}

