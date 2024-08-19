// File: src/cuda_integration/SOP_CudaIntegration.h

#ifndef __SOP_CudaIntegration_h__
#define __SOP_CudaIntegration_h__

#include <SOP/SOP_Node.h>

class SOP_CudaIntegration : public SOP_Node {
public:
    static OP_Node* myConstructor(OP_Network*, const char*, OP_Operator*);
    static PRM_Template myTemplateList[];

protected:
    SOP_CudaIntegration(OP_Network *net, const char *name, OP_Operator *op);
    virtual ~SOP_CudaIntegration();

    virtual OP_ERROR cookMySop(OP_Context &context) override;

private:
    void runCudaAdvection(float* posX, float* posY, float* posZ,
                          const float* velX, const float* velY, const float* velZ,
                          int numParticles, float deltaTime);
};

#endif
