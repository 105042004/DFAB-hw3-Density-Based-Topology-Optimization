#ifndef GRIDGENERATION_HH
#define GRIDGENERATION_HH

#include <Eigen/Sparse>
#include <iostream>

inline void generateGrid(const Eigen::Vector3d &dimensions, const Eigen::Vector3i &samplesPerDim, Eigen::MatrixX3d &V, Eigen::MatrixX4i &T, Eigen::VectorXi &voxelForTet) {
    if ((samplesPerDim.array() < 2).any()) { throw std::runtime_error("There should be at least 2 samples in nonempty dimensions (and exactly 1 in empty dimensions)"); }

    int numVertices = samplesPerDim.prod();
    V.resize(numVertices, 3);
    int nx = samplesPerDim[0],
        ny = samplesPerDim[1],
        nz = samplesPerDim[2];
    Eigen::Vector3d cellDim = dimensions.array() / (samplesPerDim.cast<double>().array() - 1.0);
    auto vtx_idx = [&](int i, int j, int k) { return i + j * nx + k * (nx * ny); }; // 1d vertex index
    auto vox_idx = [&](int i, int j, int k) { return i + j * (nx - 1) + k * (nx - 1) * (ny - 1); }; // 1d voxel index

    int numVoxels = (samplesPerDim.array() - 1).prod();
    T.resize(5 * numVoxels, 4); // we divide each voxel element into 5 tetrahedra.
    voxelForTet.resize(5 * numVoxels);

    // Cube and tet node numbering convention:
    // 3,_________,2
    //  |\        |\             3
    //  | 7---------6            *
    //  | |       | |           / \`.
    // 0|_|_______|1|          /   \ `* 2
    //  \ |       \ |         / _.--\ /
    //   \|        \|       0*-------* 1
    //    +---------+
    //   4           5
    int offset = 0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                V.row(vtx_idx(i, j, k)) << i * cellDim[0], j * cellDim[1], k * cellDim[2];
                if ((i == nx - 1) || (j == ny - 1) || (k == nz - 1)) continue;
                std::array<int, 8> cube = {{ vtx_idx(i    , j    , k    ),
                                             vtx_idx(i + 1, j    , k    ),
                                             vtx_idx(i + 1, j + 1, k    ),
                                             vtx_idx(i    , j + 1, k    ),
                                             vtx_idx(i    , j    , k + 1),
                                             vtx_idx(i + 1, j    , k + 1),
                                             vtx_idx(i + 1, j + 1, k + 1),
                                             vtx_idx(i    , j + 1, k + 1) }};

                T.block<5, 4>(offset, 0) <<
                       cube[0], cube[4], cube[5], cube[7], // tetrahedron at corner 4 (front-bottom-left)
                       cube[5], cube[6], cube[2], cube[7], // tetrahedron at corner 6 (front-top-right)
                       cube[0], cube[5], cube[1], cube[2], // tetrahedron at corner 1 (back-bottom-right)
                       cube[2], cube[3], cube[0], cube[7], // tetrahedron at corner 3 (back-top-left)
                       cube[0], cube[7], cube[5], cube[2]; // interior regular tetrahedron
                voxelForTet.segment<5>(offset).setConstant(vox_idx(i, j, k));
                offset += 5;
            }
        }
    }
}

#endif /* end of include guard: GRIDGENERATION_HH */
