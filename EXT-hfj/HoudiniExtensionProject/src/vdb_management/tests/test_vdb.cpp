// File: src/vdb_management/tests/test_vdb.cpp

#include <iostream>
#include <assert.h>
#include "../SOP_VDBManagement.h"
#include <openvdb/openvdb.h>

void testVDBProcessing() {
    // Initialize OpenVDB
    openvdb::initialize();

    // Create a simple VDB grid
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0);

    // Activate some voxels
    for (int x = -10; x < 10; ++x) {
        for (int y = -10; y < 10; ++y) {
            for (int z = -10; z < 10; ++z) {
                grid->setValue(openvdb::Coord(x, y, z), 1.0f);
            }
        }
    }

    // Process the grid using SOP_VDBManagement logic
    SOP_VDBManagement vdbManager(nullptr, "test", nullptr);
    vdbManager.processVDB(grid);

    // Validate that the processing logic worked
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        assert(*iter == 2.0f);  // All values should now be 2.0
    }

    std::cout << "All VDB tests passed!" << std::endl;
}

int main() {
    testVDBProcessing();
    return 0;
}
