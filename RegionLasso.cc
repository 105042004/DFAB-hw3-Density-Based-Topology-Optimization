#include "RegionLasso.hh"
#include <igl/AABB.h>
#include <igl/readOBJ.h>
#include <igl/facet_components.h>
#include <igl/per_vertex_normals.h>

RegionLasso::RegionLasso(const std::string &path) {
    MX3d V;
    MX3i F;
    igl::readOBJ(path, V, F);
    m_init(V, F);
}

RegionLasso::RegionLasso(const MX3d &V, const MX3i &F) {
    m_init(V, F);
}

Eigen::VectorXi RegionLasso::regionIndicesForPoints(Eigen::Ref<const MX3d> P) const {
    // We determine which region (if any) each query point is in by
    // finding its closest region boundary point and determining
    // whether the query point is on the negative side of the tangent plane.

    Eigen::VectorXd dist; // distance
    Eigen::VectorXi idx;  // index of closest triangle
    MX3d C;               // closest point
    m_aabb->squared_distance(m_V, m_F, P, dist, idx, C);

    // Get the normal at the closest point to define the tangent plane.
    MX3d N(P.rows(), 3);
    for (int i = 0; i < P.rows(); ++i) {
        // Get barycentric coords within closest triangle.
        Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor, 1, /* MaxCols */ 3> baryCoords(1, m_F.cols());
        {
            double dist_dummy;
            Eigen::Matrix<double, 1, 3> pt_dummy;
            igl::point_simplex_squared_distance<3>(C.row(i), m_V, m_F, idx[i], dist_dummy, pt_dummy, baryCoords);
        }
        // We must use an interpolated per-vertex normal to properly handle the case where the closest point
        // is a vertex or edge of the region boundary mesh; otherwise we can
        // get an incorrect inside/outside classification for points  the
        // "wrong" closest triangle is chosen.
        N.row(i) = (m_VN.row(m_F(idx[i], 0)) * baryCoords[0]
                  + m_VN.row(m_F(idx[i], 1)) * baryCoords[1]
                  + m_VN.row(m_F(idx[i], 2)) * baryCoords[2]).normalized();
    }

    Eigen::VectorXi regionIdx(P.rows());
    for (int i = 0; i < P.rows(); ++i)
        regionIdx[i] = m_regionForTri(idx[i]);
    // Overwrite region index with -1 for vertices outside the region boundaries.
    regionIdx = (((P - C).array() * N.array()).rowwise().sum() <= 0).select(regionIdx, -1);

    // Verify that all regions are matched
    std::vector<bool> seen(numRegions());
    for (int i = 0; i < P.rows(); ++i)
        if (regionIdx[i] > -1) seen[regionIdx[i]] = true;

    for (int i = 0; i < numRegions(); ++i) {
        if (!seen[i]) {
            std::cerr << "WARNING: Region " << i << " contains no query points!" << std::endl;
        }
    }

    return regionIdx;
}

void RegionLasso::m_init(const MX3d &V, const MX3i &F) {
    m_V = V;
    m_F = F;
    igl::per_vertex_normals(m_V, m_F, m_VN);
    m_aabb = std::make_unique<igl::AABB<MX3d, 3>>();
    m_aabb->init(m_V, m_F);

    igl::facet_components(m_F, m_regionForTri);
    m_numRegions = m_regionForTri.maxCoeff() + 1;
}

RegionLasso::~RegionLasso() { }
