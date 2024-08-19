// File: src/vdb_management/SOP_VDBManagement.cpp

#include "SOP_VDBManagement.h"
#include <GU/GU_Detail.h>
#include <OP/OP_Operator.h>
#include <PRM/PRM_Template.h>
#include <UT/UT_Vector3.h>
#include <GA/GA_Handle.h>

static PRM_Template theTemplates[] = {
    PRM_Template()
};

OP_Node* SOP_VDBManagement::myConstructor(OP_Network *net, const char *name, OP_Operator *op) {
    return new SOP_VDBManagement(net, name, op);
}

SOP_VDBManagement::SOP_VDBManagement(OP_Network *net, const char *name, OP_Operator *op)
    : SOP_Node(net, name, op) {}

SOP_VDBManagement::~SOP_VDBManagement() {}

OP_ERROR SOP_VDBManagement::cookMySop(OP_Context &context) {
    // Lock the SOP's geometry for writing
    gdp->clearAndDestroy();

    // Initialize OpenVDB
    openvdb::initialize();

    // Create a VDB grid with a default background value of 0.0
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0);

    // Process the VDB grid (this is where the custom logic goes)
    processVDB(grid);

    // Convert VDB grid to Houdini geometry and add to the SOP
    for (openvdb::FloatGrid::ValueOnIter iter = grid->beginValueOn(); iter; ++iter) {
        const openvdb::Coord coord = iter.getCoord();
        const float value = *iter;

        // Convert VDB coordinate to Houdini point
        UT_Vector3F pos(coord.x(), coord.y(), coord.z());
        GA_Offset ptoff = gdp->appendPoint();
        gdp->setPos3(ptoff, pos);

        // Set the value as an attribute on the point
        gdp->setFloat(ptoff, GA_ATTRIB_CD, value);
    }

    return error();
}

void SOP_VDBManagement::processVDB(openvdb::FloatGrid::Ptr grid) {
    // Example processing: Increase the value of all active voxels by 1.0
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        iter.setValue(iter.getValue() + 1.0f);
    }
}
