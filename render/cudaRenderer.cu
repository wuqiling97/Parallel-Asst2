#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

using std::cout; using std::endl;

#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
	  fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
		cudaGetErrorString(code), file, line);
	  if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif


////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

	SceneName sceneName;

	int numCircles;
	float* position;
	float* velocity;
	float* color;
	float* radius;

	int imageWidth;
	int imageHeight;
	float* imageData;
};

struct BoundingBox {
	int minX, maxX, minY, maxY;
	int width, height;
	int pixelnum;

	__host__ __device__ BoundingBox() {}

	__host__ __device__ BoundingBox(int minX, int maxX, int minY, int maxY, int w, int h, int pixelnum):
		minX(minX), maxX(maxX), minY(minY), maxY(maxY), 
		width(w), height(h), pixelnum(pixelnum) {}

	friend std::ostream& operator<<(std::ostream& out, const BoundingBox& box) {
		out<<"min, max X, Y  "<<box.minX<<' '<<box.maxX<<' '<<box.minY<<' '<<box.maxY;
		return out;
	}
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	int width = cuConstRendererParams.imageWidth;
	int height = cuConstRendererParams.imageHeight;

	if (imageX >= width || imageY >= height)
		return;

	int offset = 4 * (imageY * width + imageX);
	float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
	float4 value = make_float4(shade, shade, shade, 1.f);

	// write to global memory: As an optimization, I use a float4
	// store, that results in more efficient code than if I coded this
	// up as four seperate fp32 stores.
	*(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	int width = cuConstRendererParams.imageWidth;
	int height = cuConstRendererParams.imageHeight;

	if (imageX >= width || imageY >= height)
		return;

	int offset = 4 * (imageY * width + imageX);
	float4 value = make_float4(r, g, b, a);

	// write to global memory: As an optimization, I use a float4
	// store, that results in more efficient code than if I coded this
	// up as four seperate fp32 stores.
	*(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
	const float dt = 1.f / 60.f;
	const float pi = 3.14159;
	const float maxDist = 0.25f;

	float* velocity = cuConstRendererParams.velocity;
	float* position = cuConstRendererParams.position;
	float* radius = cuConstRendererParams.radius;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= cuConstRendererParams.numCircles)
		return;

	if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update
		return;
	}

	// determine the fire-work center/spark indices
	int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
	int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

	int index3i = 3 * fIdx;
	int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
	int index3j = 3 * sIdx;

	float cx = position[index3i];
	float cy = position[index3i+1];

	// update position
	position[index3j] += velocity[index3j] * dt;
	position[index3j+1] += velocity[index3j+1] * dt;

	// fire-work sparks
	float sx = position[index3j];
	float sy = position[index3j+1];

	// compute vector from firework-spark
	float cxsx = sx - cx;
	float cysy = sy - cy;

	// compute distance from fire-work
	float dist = sqrt(cxsx * cxsx + cysy * cysy);
	if (dist > maxDist) { // restore to starting position
		// random starting position on fire-work's rim
		float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
		float sinA = sin(angle);
		float cosA = cos(angle);
		float x = cosA * radius[fIdx];
		float y = sinA * radius[fIdx];

		position[index3j] = position[index3i] + x;
		position[index3j+1] = position[index3i+1] + y;
		position[index3j+2] = 0.0f;

		// travel scaled unit length
		velocity[index3j] = cosA/5.0;
		velocity[index3j+1] = sinA/5.0;
		velocity[index3j+2] = 0.0f;
	}
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= cuConstRendererParams.numCircles)
		return;

	float* radius = cuConstRendererParams.radius;

	float cutOff = 0.5f;
	// place circle back in center after reaching threshold radisus
	if (radius[index] > cutOff) {
		radius[index] = 0.02f;
	} else {
		radius[index] += 0.01f;
	}
}


// kernelAdvanceBouncingBalls
//
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() {
	const float dt = 1.f / 60.f;
	const float kGravity = -2.8f; // sorry Newton
	const float kDragCoeff = -0.8f;
	const float epsilon = 0.001f;

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= cuConstRendererParams.numCircles)
		return;

	float* velocity = cuConstRendererParams.velocity;
	float* position = cuConstRendererParams.position;

	int index3 = 3 * index;
	// reverse velocity if center position < 0
	float oldVelocity = velocity[index3+1];
	float oldPosition = position[index3+1];

	if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition
		return;
	}

	if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball
		velocity[index3+1] *= kDragCoeff;
	}

	// update velocity: v = u + at (only along y-axis)
	velocity[index3+1] += kGravity * dt;

	// update positions (only along y-axis)
	position[index3+1] += velocity[index3+1] * dt;

	if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
		&& oldPosition < 0.0f
		&& fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball
		velocity[index3+1] = 0.f;
		position[index3+1] = 0.f;
	}
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= cuConstRendererParams.numCircles)
		return;

	const float dt = 1.f / 60.f;
	const float kGravity = -1.8f; // sorry Newton
	const float kDragCoeff = 2.f;

	int index3 = 3 * index;

	float* positionPtr = &cuConstRendererParams.position[index3];
	float* velocityPtr = &cuConstRendererParams.velocity[index3];

	// loads from global memory
	float3 position = *((float3*)positionPtr);
	float3 velocity = *((float3*)velocityPtr);

	// hack to make farther circles move more slowly, giving the
	// illusion of parallax
	float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

	// add some noise to the motion to make the snow flutter
	float3 noiseInput;
	noiseInput.x = 10.f * position.x;
	noiseInput.y = 10.f * position.y;
	noiseInput.z = 255.f * position.z;
	float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
	noiseForce.x *= 7.5f;
	noiseForce.y *= 5.f;

	// drag
	float2 dragForce;
	dragForce.x = -1.f * kDragCoeff * velocity.x;
	dragForce.y = -1.f * kDragCoeff * velocity.y;

	// update positions
	position.x += velocity.x * dt;
	position.y += velocity.y * dt;

	// update velocities
	velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
	velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

	float radius = cuConstRendererParams.radius[index];

	// if the snowflake has moved off the left, right or bottom of
	// the screen, place it back at the top and give it a
	// pseudorandom x position and velocity.
	if ( (position.y + radius < 0.f) ||
		 (position.x + radius) < -0.f ||
		 (position.x - radius) > 1.f)
	{
		noiseInput.x = 255.f * position.x;
		noiseInput.y = 255.f * position.y;
		noiseInput.z = 255.f * position.z;
		noiseForce = cudaVec2CellNoise(noiseInput, index);

		position.x = .5f + .5f * noiseForce.x;
		position.y = 1.35f + radius;

		// restart from 0 vertical velocity.  Choose a
		// pseudo-random horizontal velocity.
		velocity.x = 2.f * noiseForce.y;
		velocity.y = 0.f;
	}

	// store updated positions and velocities to global memory
	*((float3*)positionPtr) = position;
	*((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) 
{
	float diffX = p.x - pixelCenter.x;
	float diffY = p.y - pixelCenter.y;
	float pixelDist = diffX * diffX + diffY * diffY;
	float rad = cuConstRendererParams.radius[circleIndex];

	float3 rgb;
	float alpha;

	// there is a non-zero contribution.  Now compute the shading value

	// This conditional is in the inner loop, but it evaluates the
	// same direction for all threads so it's cost is not so
	// bad. Attempting to hoist this conditional is not a required
	// student optimization in Assignment 2
	if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

		const float kCircleMaxAlpha = .5f;
		const float falloffScale = 4.f;

		float normPixelDist = sqrt(pixelDist) / rad;
		rgb = lookupColor(normPixelDist);

		float maxAlpha = .6f + .4f * (1.f-p.z);
		maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
		alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

	} else {
		// simple: each circle has an assigned color
		int index3 = 3 * circleIndex;
		rgb = *(float3*)&(cuConstRendererParams.color[index3]);
		alpha = .5f;
	}

	float oneMinusAlpha = 1.f - alpha;

	// BEGIN SHOULD-BE-ATOMIC REGION
	// global memory read

	float4 existingColor = *imagePtr;
	float4 newColor;
	newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
	newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
	newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
	newColor.w = alpha + existingColor.w;

	// global memory write
	*imagePtr = newColor;

	// END SHOULD-BE-ATOMIC REGION
}

__global__ void kernelGetBBox(BoundingBox* bound_box)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= cuConstRendererParams.numCircles)
		return;

	int index3 = 3 * index;

	// read position and radius
	float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
	float  rad = cuConstRendererParams.radius[index];

	// compute the bounding box of the circle. The bound is in integer
	// screen coordinates, so it's clamped to the edges of the screen.
	short imageWidth = cuConstRendererParams.imageWidth;
	short imageHeight = cuConstRendererParams.imageHeight;
	short minX = static_cast<short>(imageWidth * (p.x - rad));
	short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
	short minY = static_cast<short>(imageHeight * (p.y - rad));
	short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

	// a bunch of clamps.  Is there a CUDA built-in for this?
	short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
	short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
	short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
	short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;
	int width = screenMaxX - screenMinX;
	int height = screenMaxY - screenMinY;
	int pixelnum = width * height;

	bound_box[index] = BoundingBox(
		screenMinX, screenMaxX, screenMinY, screenMaxY, width, height, pixelnum
	);
}

__global__ void kernelGetPixelCricleNum(int* pixel_circlenum, int2 topleft, int circleIndex)
{
	int pixelX = blockDim.x * blockIdx.x + threadIdx.x + topleft.x;
	int pixelY = blockDim.y * blockIdx.y + threadIdx.y + topleft.y;
	if(pixelX >= cuConstRendererParams.imageWidth || pixelY >= cuConstRendererParams.imageHeight)
		return;

	float3 circlePos = *(float3*)(&cuConstRendererParams.position[3*circleIndex]);
	const int width = cuConstRendererParams.imageWidth;
	const int height = cuConstRendererParams.imageHeight;

	int pixelIdx = pixelY * width + pixelX;
	float pXcenter = float(pixelX + 0.5)/width;
	float pYcenter = float(pixelY + 0.5)/height;
	float diffX = pXcenter - circlePos.x;
	float diffY = pYcenter - circlePos.y;
	float distance = diffX*diffX + diffY*diffY;
	float radius = cuConstRendererParams.radius[circleIndex];

	if(distance <= radius*radius)
		atomicAdd(pixel_circlenum + pixelIdx, 1);

	// if(circleIndex==0 && pixelY==512 && pixelX%10==0) {
	// 	printf("(%d, %d), diffxy: %f %f  radius: %f\n", pixelX, pixelY, diffX, diffY, radius);
	// }
}

__global__ void kernelGetPixelCricleList(
	int* pixel_circle_list, int* pixel_list_ptr, 
	int2 topleft, int circleIndex, int max_pixel_circlenum
)
{
	int pixelX = blockDim.x * blockIdx.x + threadIdx.x + topleft.x;
	int pixelY = blockDim.y * blockIdx.y + threadIdx.y + topleft.y;
	if(pixelX >= cuConstRendererParams.imageWidth || pixelY >= cuConstRendererParams.imageHeight)
		return;

	float3 circlePos = *(float3*)(&cuConstRendererParams.position[3*circleIndex]);
	const int width = cuConstRendererParams.imageWidth;
	const int height = cuConstRendererParams.imageHeight;

	int pixelIdx = pixelY * width + pixelX;
	float pXcenter = float(pixelX + 0.5)/width;
	float pYcenter = float(pixelY + 0.5)/height;
	float diffX = pXcenter - circlePos.x;
	float diffY = pYcenter - circlePos.y;
	float distance = diffX*diffX + diffY*diffY;
	float radius = cuConstRendererParams.radius[circleIndex];

	if(distance <= radius*radius) {
		//inside
		// get old and atomic update list ptr
		int list_idx = max_pixel_circlenum * pixelIdx + 
		               atomicAdd(pixel_list_ptr + pixelIdx, 1);
		pixel_circle_list[list_idx] = circleIndex;
	}
}

__global__ void kernelSortPixel(
	int* pixel_circle_list, int* pixel_circlenum, int max_pixel_circlenum
)
{
	int pixelX = blockDim.x * blockIdx.x + threadIdx.x;
	int pixelY = blockDim.y * blockIdx.y + threadIdx.y;
	if(pixelX >= cuConstRendererParams.imageWidth || 
	   pixelY >= cuConstRendererParams.imageHeight)
		return;

	const int width = cuConstRendererParams.imageWidth;
	const int height = cuConstRendererParams.imageHeight;
	int pixelIdx = pixelY * width + pixelX;
	int circle_count = pixel_circlenum[pixelIdx];
	int list_start = max_pixel_circlenum * pixelIdx;
	int* thislist = pixel_circle_list + list_start;

	thrust::sort(thrust::device, thislist, thislist + circle_count);
}

__global__ void kernelGetPixelColor(
	int* pixel_circle_list, int* pixel_circlenum, int max_pixel_circlenum
)
{
	int pixelX = blockDim.x * blockIdx.x + threadIdx.x;
	int pixelY = blockDim.y * blockIdx.y + threadIdx.y;
	if(pixelX >= cuConstRendererParams.imageWidth || 
	   pixelY >= cuConstRendererParams.imageHeight)
		return;

	const int width = cuConstRendererParams.imageWidth;
	const int height = cuConstRendererParams.imageHeight;
	int pixelIdx = pixelY * width + pixelX;
	int circle_count = pixel_circlenum[pixelIdx];
	int list_start = max_pixel_circlenum * pixelIdx;
	int* thislist = pixel_circle_list + list_start;

	float pXcenter = float(pixelX + 0.5)/width;
	float pYcenter = float(pixelY + 0.5)/height;

	float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * pixelIdx]);
	for(int i=0; i<circle_count; i++) {
		int circleIdx = thislist[i];
		float3 pos = *(float3*)(&cuConstRendererParams.position[3*circleIdx]);
		shadePixel(circleIdx, make_float2(pXcenter, pYcenter), pos, imgPtr);
	}
}



////////////////////////////////////////////////////////////////////////////////////////

// caculate the unit number needed to do the task
inline int unitcount(int tasksize, int unitsize)
{
	return (tasksize + unitsize - 1)/unitsize;
}

template<typename T>
T cudaThrustMax(T* arr, int length)
{
	thrust::device_ptr<T> dev_arr(arr);
	T ret = *thrust::max_element(dev_arr, dev_arr + length);
	return ret;
}

template<typename T>
void cudaThrustSort(T* begin, T* end)
{
	thrust::device_ptr<T> dev_begin(begin), dev_end(end);
	thrust::sort(dev_begin, dev_end);
}

void
CudaRenderer::render() 
{
	// 256 threads per block is a healthy number
	const dim3 blockDim(256, 1);
	dim3 gridDim(unitcount(numCircles, blockDim.x));
	const int tot_pixelnum = image->width * image->height;

	// 计算bounding box, 并传回host
	BoundingBox* dev_bound_box;
	BoundingBox* bound_box;
	bound_box = new BoundingBox[numCircles];
	cudaMalloc(&dev_bound_box, numCircles * sizeof(BoundingBox));

	kernelGetBBox<<<gridDim, blockDim>>>(dev_bound_box);
	cudaCheckError(cudaDeviceSynchronize());
	cudaMemcpy(bound_box, dev_bound_box, numCircles * sizeof(BoundingBox), cudaMemcpyDeviceToHost);

	//right
	// for(int i=0; i<numCircles; i++) {
	// 	cout<<i<<' '<<bound_box[i]<<endl;
	// }

	// record every pixel's circle number
	int* dev_pixel_circlenum;
	cudaMalloc(&dev_pixel_circlenum, tot_pixelnum * sizeof(int));
	cudaMemset(dev_pixel_circlenum, 0, tot_pixelnum * sizeof(int));

	// 统计每个像素上圆的数量
	for(int i=0; i<numCircles; i++) {
		const dim3 blockDim(16, 16);
		const BoundingBox& box = bound_box[i];
		dim3 gridDim(unitcount(box.width, blockDim.x), unitcount(box.height, blockDim.y));
		kernelGetPixelCricleNum<<<gridDim, blockDim>>>(
			dev_pixel_circlenum, make_int2(box.minX, box.minY), i
		);
	}
	cudaCheckError(cudaDeviceSynchronize());

	int* pixel_circlenum = new int[tot_pixelnum];
	cudaMemcpy(pixel_circlenum, dev_pixel_circlenum, 
		       tot_pixelnum * sizeof(int), cudaMemcpyDeviceToHost);

	// for(int x=0; x<image->width; x++) {
	// 	for(int y=0; y<image->height; y++) {
	// 		if(y==512 && x%20==0) {
	// 			printf("(%d, %d), circle num : %d\n", x, y, pixel_circlenum[y*image->width + x]);
	// 		}
	// 	}
	// }


	// 分配每个像素存储圆编号的内存
	int* dev_pixel_circle_list;
	int max_pixel_circlenum = cudaThrustMax(dev_pixel_circlenum, tot_pixelnum);
	cudaMalloc(&dev_pixel_circle_list, sizeof(int)*max_pixel_circlenum*tot_pixelnum);

	int* dev_pixel_list_ptr; //当前圆编号列表大小
	cudaMalloc(&dev_pixel_list_ptr, sizeof(int)*tot_pixelnum);
	cudaCheckError(cudaMemset(dev_pixel_list_ptr, 0, sizeof(int)*tot_pixelnum));

	printf("get list begin\n");

	// 获得每个像素上圆的列表(无序)
	for(int i=0; i<numCircles; i++) {
		const dim3 blockDim(16, 16);
		const BoundingBox& box = bound_box[i];
		dim3 gridDim(unitcount(box.width, blockDim.x), unitcount(box.height, blockDim.y));
		kernelGetPixelCricleList<<<gridDim, blockDim>>>(
			dev_pixel_circle_list, dev_pixel_list_ptr, 
			make_int2(box.minX, box.minY), i, max_pixel_circlenum
		);
	}
	cudaCheckError(cudaDeviceSynchronize())

	printf("get list finished\n");

	// for(int i=0, j=0; i<tot_pixelnum; i++) {
	// 	cudaThrustSort(dev_pixel_circle_list + j, dev_pixel_circle_list + j + pixel_circlenum[i]);
	// 	j += max_pixel_circlenum;
	// }
	const dim3 blockDim2(16, 16);
	const dim3 gridDim2(unitcount(image->width, blockDim2.x), unitcount(image->height, blockDim2.y));
	kernelSortPixel<<<gridDim2, blockDim2>>>(
		dev_pixel_circle_list, dev_pixel_circlenum, max_pixel_circlenum);

	cudaCheckError(cudaDeviceSynchronize());

	printf("sort finished\n");

	// const dim3 blockDim2(16, 16);
	// const dim3 gridDim2(unitcount(image->width, blockDim2.x), unitcount(image->height, blockDim2.y));
	kernelGetPixelColor<<<gridDim2, blockDim2>>>(
		dev_pixel_circle_list, dev_pixel_circlenum, max_pixel_circlenum);
	cudaCheckError(cudaDeviceSynchronize());

	printf("all finished\n");

	delete[] bound_box;
	delete[] pixel_circlenum;
	cudaFree(dev_pixel_circlenum);
	cudaFree(dev_bound_box);
	cudaFree(dev_pixel_circle_list);
	cudaFree(dev_pixel_list_ptr);
}

CudaRenderer::CudaRenderer() {
	image = NULL;

	numCircles = 0;
	position = NULL;
	velocity = NULL;
	color = NULL;
	radius = NULL;

	cudaDevicePosition = NULL;
	cudaDeviceVelocity = NULL;
	cudaDeviceColor = NULL;
	cudaDeviceRadius = NULL;
	cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

	if (image) {
		delete image;
	}

	if (position) {
		delete [] position;
		delete [] velocity;
		delete [] color;
		delete [] radius;
	}

	if (cudaDevicePosition) {
		cudaFree(cudaDevicePosition);
		cudaFree(cudaDeviceVelocity);
		cudaFree(cudaDeviceColor);
		cudaFree(cudaDeviceRadius);
		cudaFree(cudaDeviceImageData);
	}
}

const Image*
CudaRenderer::getImage() {

	// need to copy contents of the rendered image from device memory
	// before we expose the Image object to the caller

	printf("Copying image data from device\n");

	cudaMemcpy(image->data,
			   cudaDeviceImageData,
			   sizeof(float) * 4 * image->width * image->height,
			   cudaMemcpyDeviceToHost);

	return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
	sceneName = scene;
	loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

	int deviceCount = 0;
	std::string name;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);

	printf("---------------------------------------------------------\n");
	printf("Initializing CUDA for CudaRenderer\n");
	printf("Found %d CUDA devices\n", deviceCount);

	for (int i=0; i<deviceCount; i++) {
		cudaDeviceProp deviceProps;
		cudaGetDeviceProperties(&deviceProps, i);
		name = deviceProps.name;

		printf("Device %d: %s\n", i, deviceProps.name);
		printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
		printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
		printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
	}
	printf("---------------------------------------------------------\n");

	// By this time the scene should be loaded.  Now copy all the key
	// data structures into device memory so they are accessible to
	// CUDA kernels
	//
	// See the CUDA Programmer's Guide for descriptions of
	// cudaMalloc and cudaMemcpy

	cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
	cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
	cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
	cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
	cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

	cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

	// Initialize parameters in constant memory.  We didn't talk about
	// constant memory in class, but the use of read-only constant
	// memory here is an optimization over just sticking these values
	// in device global memory.  NVIDIA GPUs have a few special tricks
	// for optimizing access to constant memory.  Using global memory
	// here would have worked just as well.  See the Programmer's
	// Guide for more information about constant memory.

	GlobalConstants params;
	params.sceneName = sceneName;
	params.numCircles = numCircles;
	params.imageWidth = image->width;
	params.imageHeight = image->height;
	params.position = cudaDevicePosition;
	params.velocity = cudaDeviceVelocity;
	params.color = cudaDeviceColor;
	params.radius = cudaDeviceRadius;
	params.imageData = cudaDeviceImageData;

	cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

	// also need to copy over the noise lookup tables, so we can
	// implement noise on the GPU
	int* permX;
	int* permY;
	float* value1D;
	getNoiseTables(&permX, &permY, &value1D);
	cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
	cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
	cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

	// last, copy over the color table that's used by the shading
	// function for circles in the snowflake demo

	float lookupTable[COLOR_MAP_SIZE][3] = {
		{1.f, 1.f, 1.f},
		{1.f, 1.f, 1.f},
		{.8f, .9f, 1.f},
		{.8f, .9f, 1.f},
		{.8f, 0.8f, 1.f},
	};

	cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

	if (image)
		delete image;
	image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

	// 256 threads per block is a healthy number
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(
		(image->width + blockDim.x - 1) / blockDim.x,
		(image->height + blockDim.y - 1) / blockDim.y);

	if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
		kernelClearImageSnowflake<<<gridDim, blockDim>>>();
	} else {
		kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
	}
	cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
	 // 256 threads per block is a healthy number
	dim3 blockDim(256, 1);
	dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

	// only the snowflake scene has animation
	if (sceneName == SNOWFLAKES) {
		kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
	} else if (sceneName == BOUNCING_BALLS) {
		kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
	} else if (sceneName == HYPNOSIS) {
		kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
	} else if (sceneName == FIREWORKS) {
		kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
	}
	cudaDeviceSynchronize();
}

