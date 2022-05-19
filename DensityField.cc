#include "DensityField.hh"

// Note: the indexing here must be compatible with `vox_idx` in `GridGeneration.hh`
inline int vox_idx(const Eigen::Vector3i &coords, const Eigen::Vector3i &gridShape) {
    return coords[0] + coords[1] * gridShape[0] + coords[2] * (gridShape[0] * gridShape[1]);
}

inline Eigen::Vector3i getCoords(int i, const Eigen::Vector3i &gridShape) {
    Eigen::Vector3i coords;
    coords[0] = i % gridShape[0]; i /= gridShape[0];
    coords[1] = i % gridShape[1]; i /= gridShape[1];
    coords[2] = i % gridShape[2]; i /= gridShape[2];
    return coords;
}

inline int getNeighbor(Eigen::Vector3i coords, int dim, int dir, const Eigen::Vector3i &gridShape) {
    coords[dim] += dir;
    if ((coords[dim] < 0) || (coords[dim] >= gridShape[dim]))
        return -1;
    return vox_idx(coords, gridShape);
}

inline int getNeighbor(Eigen::Vector3i coords, const Eigen::Vector3i &offset, const Eigen::Vector3i &gridShape) {
    coords += offset;
    if ((coords.array() < 0).any() || (coords.array() >= gridShape.array()).any())
        return -1;
    return vox_idx(coords, gridShape);
}

inline int getNeighbor(int i, int dim, int dir, const Eigen::Vector3i &gridShape) {
    return getNeighbor(getCoords(i, gridShape), dim, dir, gridShape);
}

////////////////////////////////////////////////////////////////////////////////
// VoxelSmoothingFilter
////////////////////////////////////////////////////////////////////////////////
VoxelSmoothingFilter::VoxelSmoothingFilter(int numVoxels, Eigen::Vector3i gridShape) {
    // Build the matrix "A" that performs an unweighted average with all voxels in
    // the surrounding neighborhood.
    std::vector<int> stencilMembers;

    // TODO: Task 3.7
    // Update this method to construct the unweighted-averaging operator as
    // sparse matrix `A`.

    for (int i = 0; i < numVoxels; ++i) {
        stencilMembers.assign(1, i); // Include self

        Eigen::Vector3i coords = getCoords(i, gridShape);
        Eigen::Vector3i d;
        // Include all neighbors in the 3x3x3 cube centered at i.
        for (d[0] = -1; d[0] <= 1; ++d[0]) {
            for (d[1] = -1; d[1] <= 1; ++d[1]) {
                for (d[2] = -1; d[2] <= 1; ++d[2]) {
                    if ((d.array() == 0).all()) continue;
                    int n = getNeighbor(coords, d, gridShape);
                    if (n >= 0) stencilMembers.push_back(n);
                }
            }
        }
    }

    A.resize(numVoxels, numVoxels);
    A.setIdentity();
}

////////////////////////////////////////////////////////////////////////////////
// Self-Supporting Filter
////////////////////////////////////////////////////////////////////////////////
// Get `cell`'s neighbors in its + shaped stencil in the layer below 
void VoxelSelfSupportingFilter::getSupportingNeighbors(const Eigen::Vector3i &cell, std::vector<int> &supportingNeighbors) const {
    supportingNeighbors.clear();
    // TODO: Bonus
}

DensityField::VXd VoxelSelfSupportingFilter::apply(Eigen::Ref<const VXd> x) {
    m_input_x = x;    // Note: the input variables be saved for use in backprop!
    m_filtered_x = x; // Note: the final output variables must also be saved for use in backprop!
    std::vector<int> supportingNeighbors;
    // From bottom to top (along y direction), set the density at a grid point
    // to be the minimum of the desired value and the maximum of the densities
    // in the supporting grid
    for (int j = 1; j < gridShape[1]; ++j) {
        for (int i = 0; i < gridShape[0]; ++i) {
            for (int k = 0; k < gridShape[2]; ++k) {
                getSupportingNeighbors(Eigen::Vector3i(i, j, k), supportingNeighbors);

                // TODO: Bonus
            }
        }
    }

    return m_filtered_x;
}

DensityField::VXd VoxelSelfSupportingFilter::backprop(Eigen::Ref<const VXd> dJ_dout) const {
    VXd dJ_din = dJ_dout;

    // TODO: Bonus

    return dJ_din;
}
