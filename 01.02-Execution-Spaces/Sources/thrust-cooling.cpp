
#include <thrust/execution_policy.h>
#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <cstdio>

int main() {
    float k = 0.5;
    float ambient_temp = 20;
    thrust::universal_vector<float> temp {42,24,50};
    //This is a way of defining a function which can run on the GPU which is not explicitly a CUDA kernel
    auto transformation = [=] __host__ __device__ (float temp) {return temp + k *(ambient_temp -temp);};

    std::printf("step temp[0] temp[1] temp[2]\n");
    for (int step = 0; step < 3; step++) {
        //This means: call the transform function from the thrust library, make it run from the device, loop from temp.begin() to temp.end() and start 
        //writing to temp.begin() (i.e. overwriting) - the function that youre using is the transformation function (which has been __device__ enabled)
        thrust::transform(thrust::device, temp.begin(), temp.end(), temp.begin(), transformation);
        std::printf("%d     %.2f    %.2f    %.2f\n", step, temp[0], temp[1], temp[2]);
    }
}
