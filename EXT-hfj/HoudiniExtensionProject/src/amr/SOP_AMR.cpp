// File: src/amr/SOP_AMR.cpp

#include "SOP_AMR.h"
#include <GU/GU_Detail.h>
#include <OP/OP_Operator.h>
#include <PRM/PRM_Template.h>
#include <UT/UT_Vector3.h>
#include <GA/GA_Iterator.h>
#include <GA/GA_Handle.h>

static PRM_Template theTemplates[] = {
    PRM_Template()
};

OP_Node* SOP_AMR::myConstructor(OP_Network *net, const char *name, OP_Operator *op) {
    return new SOP_AMR(net, name, op);
}

SOP_AMR::SOP_AMR(OP_Network *net, const char *name, OP_Operator *op)
    : SOP_Node(net, name, op) {}

SOP_AMR::~SOP_AMR() {}

OP_ERROR SOP_AMR::cookMySop(OP_Context &context) {
    // Lock geometry for writing
    gdp->clearAndDestroy();

    // Perform the adaptive mesh refinement
    refineMesh();

    return error();
}

void SOP_AMR::refineMesh() {
    // Example implementation: Refine mesh based on velocity magnitude

    GA_RWHandleV3 velAttrib(gdp->findAttribute(GA_ATTRIB_POINT, "v"));
    if (!velAttrib.isValid()) return;

    GA_RWHandleF densityAttrib(gdp->addFloatTuple(GA_ATTRIB_POINT, "density", 1));

    GA_Offset ptoff;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        UT_Vector3 vel = velAttrib.get(ptoff);
        float velocityMagnitude = vel.length();

        if (velocityMagnitude > 1.0f) {
            // Increase density in high-velocity regions
            densityAttrib.set(ptoff, 1.0f);

            // Example refinement logic (create additional points)
            for (int i = 0; i < 2; ++i) {
                GA_Offset newPtoff = gdp->appendPointOffset();
                gdp->setPos3(newPtoff, gdp->getPos3(ptoff) + UT_Vector3(i * 0.1f));
                velAttrib.set(newPtoff, vel);
                densityAttrib.set(newPtoff, 1.0f);
            }
        } else {
            densityAttrib.set(ptoff, 0.1f);  // Low density for low-velocity regions
        }
    }
}
