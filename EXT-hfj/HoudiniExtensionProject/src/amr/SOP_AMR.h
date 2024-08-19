// File: src/amr/SOP_AMR.h

#ifndef __SOP_AMR_h__
#define __SOP_AMR_h__

#include <SOP/SOP_Node.h>

class SOP_AMR : public SOP_Node {
public:
    static OP_Node* myConstructor(OP_Network*, const char*, OP_Operator*);
    static PRM_Template myTemplateList[];

protected:
    SOP_AMR(OP_Network *net, const char *name, OP_Operator *op);
    virtual ~SOP_AMR();

    virtual OP_ERROR cookMySop(OP_Context &context) override;

private:
    void refineMesh();
};

#endif
