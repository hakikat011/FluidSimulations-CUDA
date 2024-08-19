// File: src/vdb_management/SOP_VDBManagement.h

#ifndef __SOP_VDBManagement_h__
#define __SOP_VDBManagement_h__

#include <SOP/SOP_Node.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/ChangeBackground.h>

class SOP_VDBManagement : public SOP_Node {
public:
    static OP_Node* myConstructor(OP_Network*, const char*, OP_Operator*);
    static PRM_Template myTemplateList[];

protected:
    SOP_VDBManagement(OP_Network *net, const char *name, OP_Operator *op);
    virtual ~SOP_VDBManagement();

    virtual OP_ERROR cookMySop(OP_Context &context) override;

private:
    void processVDB(openvdb::FloatGrid::Ptr grid);
};

#endif
