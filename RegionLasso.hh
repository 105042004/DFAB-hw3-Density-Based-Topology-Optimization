////////////////////////////////////////////////////////////////////////////////
// RegionLasso.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Support for selecting points in a 3D scene using a collection of closed
//  triangle meshes.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  04/28/2021 21:02:31
////////////////////////////////////////////////////////////////////////////////
#ifndef REGIONLASSO_HH
#define REGIONLASSO_HH

#include <Eigen/Dense>
#include <memory>

// Forward-declare AABB data structure (to isolate `igl/aabb.h` include in .cc file).
namespace igl {
    template<typename DerivedV, int DIM>
    struct AABB;
}

struct RegionLasso {
    using MX3d = Eigen::MatrixX3d;
    using MX3i = Eigen::MatrixX3i;

    RegionLasso(const std::string &path);
    RegionLasso(const MX3d &V, const MX3i &F);

    Eigen::VectorXi regionIndicesForPoints(Eigen::Ref<const MX3d> P) const;

    int numRegions() const { return m_numRegions; }

    // Destructor must be implemented in .cc file since m_aabb is an incomplete type
    ~RegionLasso();

private:
    void m_init(const MX3d &V, const MX3i &F);

    MX3d m_V, m_VN;
    MX3i m_F;
    int m_numRegions;
    Eigen::VectorXi m_regionForTri;
    std::unique_ptr<igl::AABB<MX3d, 3>> m_aabb;
};

#endif /* end of include guard: REGIONLASSO_HH */
