#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include "timer.h"
//#include "check.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SOFTENING 1e-9f
typedef struct { float x, y, z, vx, vy, vz; } Body;
void randomizeBodies(float* data, int n) {
	for (int i = 0; i < n; i++) {
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}
}

__global__ void bodyForce(Body* p, float dt, int n) {
	int stride = blockDim.x * gridDim.x;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	for (; i < n; i += stride)
	{
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

		for (int j = 0; j < n; j++) {
			float dx = p[j].x - p[i].x;
			float dy = p[j].y - p[i].y;
			float dz = p[j].z - p[i].z;
			float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
			float invDist = rsqrtf(distSqr);
			float invDist3 = invDist * invDist * invDist;

			Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
		}

		p[i].vx += dt * Fx; p[i].vy += dt * Fy; p[i].vz += dt * Fz;
	}
}

__global__ void physics(Body* p, float dt, int n)
{
	int stride = blockDim.x * gridDim.x;

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
	{
		p[i].x += p[i].vx * dt;
		p[i].y += p[i].vy * dt;
		p[i].z += p[i].vz * dt;
	}
}

int main(const int argc, const char** argv) {

	int nBodies = 2 << 11;
	int salt = 0;
	if (argc > 1) nBodies = 2 << atoi(argv[1]);
	if (argc > 2) salt = atoi(argv[2]);

	const float dt = 0.01f; // time step
	const int nIters = 10;  // simulation iterations

	int bytes = nBodies * sizeof(Body);
	float* buf;

	int deviceId;
	cudaGetDevice(&deviceId);
	int numberOfSMs;
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	cudaMallocManaged(&buf, bytes);
	cudaMemPrefetchAsync(buf, bytes, deviceId);
	Body* p = (Body*)buf;

	randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

	double totalTime = 0.0;

	for (int iter = 0; iter < nIters; iter++) {
		StartTimer();
		bodyForce << <numberOfSMs * 32, 256 >> > (p, dt, nBodies); // compute interbody forces
		physics << <numberOfSMs * 32, 256 >> > (p, dt, nBodies);

		cudaDeviceSynchronize();
		const double tElapsed = GetTimer() / 1000.0;
		totalTime += tElapsed;
	}

	double avgTime = totalTime / (double)(nIters);
	float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

#ifdef ASSESS
	checkPerformance(buf, billionsOfOpsPerSecond, salt);
#else
	checkAccuracy(buf, nBodies);
	printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
	salt += 1;
#endif


	cudaFree(buf);
}