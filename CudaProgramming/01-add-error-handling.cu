//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//
//void _CUDACHECK(cudaError_t err)
//{
//	if (err != cudaSuccess)
//	{
//		printf("Error: %s\n", cudaGetErrorString(err));
//	}
//}
//
//void init(int* a, int N)
//{
//	int i;
//	for (i = 0; i < N; ++i)
//		a[i] = i;
//}
//
//__global__
//void doubleElements(int* a, int N)
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	int stride = gridDim.x * blockDim.x;
//	for (int i = idx; i < N + stride; i += stride)
//		a[i] *= 2;
//}
//
//bool checkElementsAreDoubled(int* a, int N)
//{
//	int i;
//	for (i = 0; i < N; ++i)
//		if (a[i] != i * 2)
//			return false;
//	return true;
//}
//
//int main()
//{
//	int deviceId;
//	cudaGetDevice(&deviceId);                  // `deviceId` now points to the id of the currently active GPU.
//
//	cudaDeviceProp props;
//	cudaGetDeviceProperties(&props, deviceId); // 'prop' is useful
//	;
//	printf("devices : %d\ndevices name : %s\n", deviceId, props.name);
//
//	int N = 10000;
//	int* a;
//
//	size_t size = N * sizeof(int);
//	_CUDACHECK(cudaMallocManaged(&a, size));
//
//	init(a, N);
//
//	//size_t threads_per_block = 2048; thread per block cannot be bigger than 1024 ! !
//	size_t threads_per_block = 1024;
//	size_t number_of_blocks = 32;
//
//	doubleElements << <number_of_blocks, threads_per_block >> > (a, N);
//
//	_CUDACHECK(cudaGetLastError());
//	cudaDeviceSynchronize();
//	_CUDACHECK(cudaGetLastError());
//
//	bool areDoubled = checkElementsAreDoubled(a, N);
//	printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");
//
//	_CUDACHECK(cudaFree(a));
//}
