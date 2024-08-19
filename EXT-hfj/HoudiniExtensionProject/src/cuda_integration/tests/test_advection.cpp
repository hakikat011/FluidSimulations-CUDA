// File: src/cuda_integration/tests/test_advection.cpp

#include <iostream>
#include <assert.h>
#include "../SOP_CudaIntegration.h"

void testAdvection() {
    // Initialize positions and velocities for testing
    const int numParticles = 10;
    float posX[numParticles] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float posY[numParticles] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float posZ[numParticles] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float velX[numParticles] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float velY[numParticles] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float velZ[numParticles] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float deltaTime = 1.0f;

    // Run the kernel
    SOP_CudaIntegration integrator;
    integrator.runCudaAdvection(posX, posY, posZ, velX, velY, velZ, numParticles, deltaTime);

    // Check if positions are updated correctly
    for (int i = 0; i < numParticles; i++) {
        assert(posX[i] == i + 1.0f); // Should have moved by 1 unit
        assert(posY[i] == i);
        assert(posZ[i] == i);
    }

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    testAdvection();
    return 0;
}
