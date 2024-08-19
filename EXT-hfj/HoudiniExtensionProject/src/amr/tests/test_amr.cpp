// File: src/amr/tests/test_amr.cpp

#include <iostream>
#include <assert.h>
#include "../SOP_AMR.h"

void testAMR() {
    // Example test for AMR logic
    // Here, you would simulate a scenario with predefined velocity vectors
    // and check that the mesh refinement behaves as expected.
    
    // Initialize Houdini's GU_Detail object (geometry container)
    GU_Detail gdp;

    // Create some initial points with velocities
    for (int i = 0; i < 5; ++i) {
        GA_Offset ptoff = gdp.appendPointOffset();
        gdp.setPos3(ptoff, UT_Vector3(i * 1.0f, 0.0f, 0.0f));
        
        GA_RWHandleV3 velAttrib(&gdp, GA_ATTRIB_POINT, "v");
        velAttrib.set(ptoff, UT_Vector3(1.0f, 0.0f, 0.0f));  // Set velocity
    }

    // Apply the AMR logic
    SOP_AMR amrNode(nullptr, "test_amr", nullptr);
    amrNode.refineMesh();

    // Validate the refinement (e.g., check the number of points)
    assert(gdp.getNumPoints() > 5);  // Should have added points

    std::cout << "All AMR tests passed!" << std::endl;
}

int main() {
    testAMR();
    return 0;
}
