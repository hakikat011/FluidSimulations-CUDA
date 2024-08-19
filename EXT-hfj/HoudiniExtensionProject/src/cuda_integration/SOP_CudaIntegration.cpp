// File: src/cuda_integration/SOP_CudaIntegration.cpp

#include "SOP_CudaIntegration.h"
#include <cuda_runtime.h>
#include <GU/GU_Detail.h>
#include <OP/OP_Operator.h>
#include <PRM/PRM_Template.h>

#include "kernels/advection.cu"

static PRM_Template theTemplates[] = {
    PRM_Template()
};

OP_Node* SOP_CudaIntegration::myConstructor(OP_Network *net, const char *name, OP_Operator *op) {
    return new SOP_CudaIntegration(net, name, op);
}

SOP_CudaIntegration::SOP_CudaIntegration(OP_Network *net, const char *name, OP_Operator *op)
    : SOP_Node(net, name, op) {}

SOP_CudaIntegration::~SOP_CudaIntegration() {}

OP_ERROR SOP_CudaIntegration::cookMySop(OP_Context &context) {
    // Lock geometry for writing
    gdp->clearAndDestroy();

    // Retrieve and prepare particle positions and velocities
    int numParticles = gdp->getNumPoints();
    float* posX = new float[numParticles];
    float* posY = new float[numParticles];
    float* posZ = new float[numParticles];
    float* velX = new float[numParticles];
    float* velY = new float[numParticles];
    float* velZ = new float[numParticles];

    // Populate arrays with data from Houdini geometry
    GA_Offset ptoff;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        const UT_Vector3 pos = gdp->getPos3(ptoff);
        const UT_Vector3 vel = gdp->getV3(ptoff, GA_ATTRIB_VELOCITY);

        posX[ptoff] = pos.x();
        posY[ptoff] = pos.y();
        posZ[ptoff] = pos.z();
        velX[ptoff] = vel.x();
        velY[ptoff] = vel.y();
        velZ[ptoff] = vel.z();
    }

    // Run CUDA kernel for advection
    runCudaAdvection(posX, posY, posZ, velX, velY, velZ, numParticles, 0.02f);

    // Write back the updated positions to Houdini geometry
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        UT_Vector3 pos(posX[ptoff], posY[ptoff], posZ[ptoff]);
        gdp->setPos3(ptoff, pos);
    }

    // Cleanup
    delete[] posX;
    delete[] posY;
    delete[] posZ;
    delete[] velX;
    delete[] velY;
    delete[] velZ;

    return error();
}

void SOP_CudaIntegration::runCudaAdvection(float* posX, float* posY, float* posZ,
                                           const float* velX, const float* velY, const float* velZ,
                                           int numParticles, float deltaTime) {

    float *d_posX, *d_posY, *d_posZ;
    float *d_velX, *d_velY, *d_velZ;

    // Allocate GPU memory
    cudaMalloc((void**)&d_posX, numParticles * sizeof(float));
    cudaMalloc((void**)&d_posY, numParticles * sizeof(float));
    cudaMalloc((void**)&d_posZ, numParticles * sizeof(float));
    cudaMalloc((void**)&d_velX, numParticles * sizeof(float));
    cudaMalloc((void**)&d_velY, numParticles * sizeof(float));
    cudaMalloc((void**)&d_velZ, numParticles * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_posX, posX, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, posY, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ, posZ, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velX, velX, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velY, velY, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velZ, velZ, numParticles * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    advectionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, numParticles, deltaTime);

    // Copy results back to CPU
    cudaMemcpy(posX, d_posX, numParticles * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(posY, d_posY, numParticles * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(posZ, d_posZ, numParticles * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_posX);
    cudaFree(d_posY);
    cudaFree(d_posZ);
    cudaFree(d_velX);
    cudaFree(d_velY);
    cudaFree(d_velZ);
}
