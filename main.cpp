
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>


#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x1000000)

int main(void) {
	// Create the two input vectors
	int i;
	const int LIST_SIZE = 1024*640;
	int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
	int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
	for (i = 0; i < LIST_SIZE; i++) {
		A[i] = i;
		B[i] = LIST_SIZE - i;
	}

	LARGE_INTEGER frequency;
	LARGE_INTEGER t1, t2;
	long long gpu_time;

	QueryPerformanceFrequency(&frequency);

	int *C_ref = (int*)malloc(sizeof(int)*LIST_SIZE);

	//DWORD fCurrentTime_s, fCurrentTime_e;
	//fCurrentTime_s = GetTickCount();
	QueryPerformanceCounter(&t1);
	for (i = 0; i < LIST_SIZE; i++)
	{
		C_ref[i] = A[i] + B[i];
	}
	QueryPerformanceCounter(&t2);
	//fCurrentTime_e = GetTickCount();

	gpu_time = (t2.QuadPart - t1.QuadPart) / (frequency.QuadPart / 100000);
	printf("CPU Time = %d us\n", gpu_time);
	
	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("VectorAdd.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	//fCurrentTime_s = GetTickCount();
	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
		&device_id, &ret_num_devices);

	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create memory buffers on the device for each vector 
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		LIST_SIZE * sizeof(int), NULL, &ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		LIST_SIZE * sizeof(int), NULL, &ret);
	cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		LIST_SIZE * sizeof(int), NULL, &ret);

	// Copy the lists A and B to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
		LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
		LIST_SIZE * sizeof(int), B, 0, NULL, NULL);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1,
		(const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

	// Execute the OpenCL kernel on the list
	QueryPerformanceCounter(&t1);
	size_t global_item_size = LIST_SIZE; // Process the entire lists
	size_t local_item_size = 64; // Divide work items into groups of 64
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		&global_item_size, &local_item_size, 0, NULL, NULL);

	QueryPerformanceCounter(&t2);
	//fCurrentTime_e = GetTickCount();
	// Read the memory buffer C on the device to the local variable C
	int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
	ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
		LIST_SIZE * sizeof(int), C, 0, NULL, NULL);

	// Display the result to the screen
	//for (i = 0; i < LIST_SIZE; i++)
	//	printf("%d + %d = %d\n", A[i], B[i], C[i]);
	

	gpu_time = (t2.QuadPart - t1.QuadPart) / (frequency.QuadPart/ 100000);
	printf("GPU Time = %d us\n", gpu_time);
	
	for (i = 0; i < LIST_SIZE; i++)
	{
		if (C_ref[i] != C[i])
		{
			printf("Fail \n");
		}
	}

	// Clean up
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(c_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(A);
	free(B);
	free(C);
	return 0;
}


//#include <cstdio>
//#include <cstdlib>
//#include <CL/opencl.h>
//
//#define IMG_HEIGHT 1024
//#define IMG_WIDTH 1024
//#define MAX_SOURCE_SIZE 1000000
//
//const char *kernel_code = "\n" \
//"__kernel void memcpy_kernel(__global unsigned char *src, __global unsigned char *dst, __private int width)\n" \
//"{\n" \
//"	int group_row_idx = get_group_id(0);\n" \
//"	int local_row_size = get_local_size(0);\n" \
//"	int local_row_idx = get_local_id(0);\n" \
//"	int group_col_idx = get_group_id(1);\n" \
//"	int local_col_size = get_local_size(1);\n" \
//"	int local_col_idx = get_local_id(1);\n" \
//"	int current_idx = (group_row_idx*local_row_size+local_row_idx)*width + (group_col_idx*local_col_size+local_col_idx);\n" \
//"	dst[current_idx] = src[current_idx];\n" \
//"}\n";
//
////int main(int argc, char *argv[]) 
//int main(void)
//{
//	//if (argc != 3) {
//	//	printf("usage : opencl_memcpy [inputfile] [outputfile]\n");
//	//	exit(1);
//	//}
//	//FILE *output_file = fopen(argv[1], "rb");
//	//FILE *output_file = fopen(argv[2], "wb");
//
//	FILE *input_file = fopen("C:/Program Files/CUDA/CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions/opencl_test_1/Debug/input_data", "rb");
//	FILE *output_file = fopen("C:/Program Files/CUDA/CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions/opencl_test_1/Debug/output_data", "wb");
//
//	unsigned char *src_buf, *dst_buf;
//
//	src_buf = (unsigned char *)malloc(sizeof(unsigned char)*IMG_HEIGHT*IMG_WIDTH);
//	dst_buf = (unsigned char *)malloc(sizeof(unsigned char)*IMG_HEIGHT*IMG_WIDTH);
//
//	for (int i = 0; i<IMG_HEIGHT*IMG_WIDTH; i++)
//		src_buf[i] = fgetc(input_file);
//	fclose(input_file);
//
//	cl_platform_id *platform_list = NULL;
//	cl_device_id *device_list = NULL;
//	cl_context context = NULL;
//	cl_command_queue command_queue = NULL;
//	cl_mem src_obj = NULL;
//	cl_mem dst_obj = NULL;
//	cl_program program = NULL;
//	cl_kernel kernel = NULL;
//	cl_uint ret_num_devices;
//	cl_uint ret_num_platforms;
//	cl_int err;
//
//	// Reference https://iws44.iiita.ac.in/wiki/opencl/doku.php?id=clgetplatformids
//
//	// Step 1 : Get platform/device information
//	if ((err = clGetPlatformIDs(0, NULL, &ret_num_platforms)) != CL_SUCCESS) { // Get # of platforms
//		printf("Error code %d\n", err); exit(1);
//	}
//	platform_list = (cl_platform_id *)malloc(ret_num_platforms * sizeof(cl_platform_id)); // Allocate mem for platform_list
//	if ((err = clGetPlatformIDs(ret_num_platforms, platform_list, NULL)) != CL_SUCCESS) { // Get platform info
//		printf("Error code %d\n", err); exit(1);
//	}
//
//	// Step 2 : Get information about the device
//	if ((err = clGetDeviceIDs(platform_list[0], CL_DEVICE_TYPE_DEFAULT, 0, NULL, &ret_num_devices)) != CL_SUCCESS) { // Get # of devices of platform 0
//		printf("Error code %d\n", err); exit(1);
//	}
//	device_list = (cl_device_id *)malloc(ret_num_devices * sizeof(cl_device_id)); // Allocate mem for device_list of platform 0
//	if ((err = clGetDeviceIDs(platform_list[0], CL_DEVICE_TYPE_DEFAULT, ret_num_devices, device_list, NULL)) != CL_SUCCESS) { // Get device info of platform 0
//		printf("Error code %d\n", err); exit(1);
//	}
//
//	// Step 3 : Create OpenCL Context
//	context = clCreateContext(NULL, ret_num_devices, &device_list[0], NULL, NULL, &err);
//	if (err != CL_SUCCESS) {
//		printf("Error code %d\n", err); exit(1);
//	}
//
//	// Step 4 : Create Command Queue
//	command_queue = clCreateCommandQueue(context, device_list[0], 0, &err);
//	if (err != CL_SUCCESS) {
//		printf("Error code %d\n", err); exit(1);
//	}
//
//	// Step 5 : Create memory objects and transfer the data to memory buffer
//	src_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, IMG_HEIGHT*IMG_WIDTH * sizeof(unsigned char), NULL, &err);
//	if (err != CL_SUCCESS) {
//		printf("Error code %d\n", err); exit(1);
//	}
//	dst_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, IMG_HEIGHT*IMG_WIDTH * sizeof(unsigned char), NULL, &err);
//	if (err != CL_SUCCESS) {
//		printf("Error code %d\n", err); exit(1);
//	}
//	if ((err = clEnqueueWriteBuffer(command_queue, src_obj, CL_TRUE, 0, IMG_HEIGHT*IMG_WIDTH * sizeof(unsigned char), (void*)src_buf, 0, NULL, NULL)) != CL_SUCCESS) {
//		printf("Error code %d\n", err); exit(1);
//	}
//
//	cl_device_type dev_type;
//	clGetDeviceInfo(0, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
//	if (dev_type == CL_DEVICE_TYPE_GPU) {
//		printf("I'm 100%% sure this device is a GPU");
//	}
//	
//	// Step 6 : Read kernel file
//	// --> Instead of reading kernel file, I put kernel source code in this file as workaround to avoid invalid UTF-8 error...
//	/*
//	FILE *kernel_fp=fopen("memcpy_kernel.cl", "r");
//	char *kernel_code = (char*)malloc(MAX_SOURCE_SIZE*sizeof(char));
//	if(kernel_fp==NULL){
//	printf("Kernel file not found\n"); exit(1);
//	}
//	size_t kernel_code_size = fread(kernel_code, sizeof(char), MAX_SOURCE_SIZE, kernel_fp);
//	fclose(kernel_fp);
//	*/
//
//	// Step 7 : Create kernel program from the string that contains kernel code
//	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_code, 0, &err);
//	
//	if (err != CL_SUCCESS) {
//		printf("Error code %d\n", err); exit(1);
//	}
//
//	// Step 8 : Build kernel program
//	if ((err = clBuildProgram(program, ret_num_devices, device_list, NULL, NULL, NULL)) != CL_SUCCESS) { // program halts here!!
//		printf("Error code %d\n", err); //exit(1);
//	}
//
//	if (err == CL_BUILD_PROGRAM_FAILURE) { // for debugging kernel code
//
//										   // Determine the size of the log
//		size_t log_size;
//		clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
//
//		// Allocate memory for the log
//		char *log = (char *)malloc(log_size * sizeof(char));
//
//		// Get the log
//		clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
//
//		// Print the log
//		printf("Kernel build log : %s\n", log);
//	}
//	if (err != CL_SUCCESS) {
//		printf("Error code %d\n", err); exit(1);
//	}
//
//	// Step 9 : Create OpenCL Kernel
//	kernel = clCreateKernel(program, "memcpy_kernel", &err);
//	if (err != CL_SUCCESS) {
//		printf("Error code %d\n", err); exit(1);
//	}
//
//	// Step 10 : Set OpenCL kernel argument
//	size_t *global_work_size;
//	size_t *global_work_offset;
//	global_work_size = (size_t *)malloc(2 * sizeof(size_t));
//	global_work_offset = (size_t *)malloc(2 * sizeof(size_t));
//	global_work_size[0] = IMG_HEIGHT; global_work_size[1] = IMG_WIDTH;
//	global_work_offset[0] = 0; global_work_offset[1] = 0;
//
//	size_t *local_work_size;
//	local_work_size = (size_t *)malloc(2 * sizeof(size_t));
//	local_work_size[0] = 16; local_work_size[1] = 16; // 16*16 work groups
//
//	cl_int width = IMG_WIDTH;
//
//	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_obj); // Kernel function memcpy_kernel's argument 0 = src_obj
//	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_obj); // Kernel function memcpy_kernel's argument 1 = dst_obj
//	err = clSetKernelArg(kernel, 2, sizeof(cl_int), &width); // Kernel function memcpy_kernel's argument 2 = width
//
//	// Step 11 : Run OpenCL kernel
//	if ((err = clEnqueueNDRangeKernel(command_queue, kernel, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL)) != CL_SUCCESS) {
//		printf("Error code %d\n", err); exit(1);
//	}
//
//	// Step 12 : Copy dst_obj to dst_buf
//
//	clEnqueueReadBuffer(command_queue, dst_obj, CL_TRUE, 0, IMG_HEIGHT*IMG_WIDTH, dst_buf, 0, NULL, NULL);
//
//	// Step 13 : Check if src_buf and dst_buf are same
//	for (int i = 0; i<IMG_HEIGHT*IMG_WIDTH; i++)
//		fputc(dst_buf[i], output_file);
//	fclose(output_file);
//
//	bool is_correct = true;
//	for (int i = 0; i<IMG_HEIGHT*IMG_WIDTH; i++) {
//		if (dst_buf[i] != src_buf[i]) {
//			is_correct = false;
//			break;
//		}
//	}
//	if (is_correct) printf("memcpy succeeded\n");
//	else printf("memcpy failed\n");
//
//	// Step 14 : Release OpenCL resources
//	clReleaseKernel(kernel);
//	clReleaseProgram(program);
//	clReleaseCommandQueue(command_queue);
//	clReleaseMemObject(src_obj);
//	clReleaseMemObject(dst_obj);
//	clReleaseContext(context);
//
//	// Step 15 : Release Host resources
//	free(src_buf);
//	free(dst_buf);
//	free(platform_list);
//	free(device_list);
//	free(global_work_size);
//	free(global_work_offset);
//	free(local_work_size);
//
//	return 0;
//}