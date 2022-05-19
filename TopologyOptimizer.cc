#include "TopologyOptimizer.hh"
#include <iostream>

#include <igl/boundary_facets.h> // for accessing the surface mesh
#include <igl/doublearea.h>
#include <igl/volume.h>

TopologyOptimizer::TopologyOptimizer(Eigen::Ref<const MX3d> V, Eigen::Ref<const MX4i> T)
    : m_V(V), m_T(T), densities(T.rows())
{
    C.setIsotropic(1000.0 /* 1 GPa */, 0.3);
    setUniformDensities();

    igl::volume(m_V, m_T, m_vols);

    Eigen::MatrixXi J, K;

    // Generate the visualization (surface) mesh
    igl::boundary_facets(m_T, m_bdryF, J, K);
    // Boundary facets are apparently oriented inward...
    for (size_t i = 0; i < m_bdryF.rows(); ++i)
        std::swap(m_bdryF(i, 0), m_bdryF(i, 1));

    igl::doublearea(m_V, m_bdryF, m_bdryAreas);
    m_bdryAreas *= 0.5;
}

TopologyOptimizer::StrainPhis TopologyOptimizer::getStrainVecPhis(int e) const {
    // TODO: Task 3.1
    // Compute the strains of the 12 vector-valued basis functions in element `e`.
    StrainPhis result;

    result[0] = Eigen::Matrix3d::Zero(); // You can compute each strain in an Eigen::Matrix3d despite the compact `SymmetricMatrix` type it's converted to.

    return result;
}

// Compute the *full density* per-element stiffness matrix.
TopologyOptimizer::PerElementStiffnessMatrix TopologyOptimizer::perElementStiffnessMatrix(int e) const {
    PerElementStiffnessMatrix Ke;
    // TODO: Task 3.2
    // Construct the 12x12 per-element stiffness matrix.
    Ke.setIdentity();

    return Ke;
}

TopologyOptimizer::SpMat TopologyOptimizer::buildStiffnessMatrix() const {
    const int numVars = 3 * numVertices();
    SpMat K(numVars, numVars);
    // TODO: Task 3.3
    // Assemble (the lower triangle of) the global stiffness matrix.
    // Remember to apply the SIMP interpolation law!
    K.setIdentity();

    return K;
}

double TopologyOptimizer::solveEquilibriumProblem(MX3d &U) const {
    // Flatten Fext in the order F0_x, F0_y, F0_z, F1_x, ....
    auto f = flatten(m_Fext.leftCols(3));
    int numVars = 3 * m_V.rows();

    if (!m_KFactorizationCache) {
        SpMat K = buildStiffnessMatrix();

        // TODO: Task 3.4
        // Modify the linear system `K u = f ` to apply the boundary conditions.

        if (!solver) { solver = std::make_unique<Solver>(); solver->analyzePattern(K); }
        solver->factorize(K);
        m_KFactorizationCache = true;
    }
    auto u = solver->solve(f).eval();

    unflatten(u, 3, U);
    return f.dot(u);
}

TopologyOptimizer::VXd TopologyOptimizer::gradCompliance(const MX3d &U) const {
    VXd dJ_drho;

    // TODO: Task 3.5
    // Compute the gradient of compliance with respect to each tet's density parameter.
    dJ_drho.setZero(numElements());

    return densities.backprop(dJ_drho);
}

TopologyOptimizer::VXd TopologyOptimizer::gradVolume() const {
    VXd dV_drho;

    // TODO: Task 3.5
    // Compute the gradient of volume with respect to each tet's density parameter.
    dV_drho.setZero(numElements());

    return densities.backprop(dV_drho);
}

void TopologyOptimizer::optimizeOC(int numSteps) {
    double m = 0.05;     // "move limit"
    double ctol = 1e-6; // tolerance for volume constraint
    double lambda_min = 1;
    double lambda_max = 2;

    double totalVolume = m_vols.sum();

    // Safeguard against abrupt changes in the target volume fraction (that cannot be satisfied due to the move limit)
    if (std::abs(maxVolumeFrac - m_vols.dot(densities.rho()) / totalVolume) > 100 * ctol)
        setUniformDensities();

    // TODO: Task 3.6
    // Optimize the densities using the Optimality Criterion Algorithm
}

SymmetricMatrixValue<double, 3> TopologyOptimizer::cauchyStress(const MX3d &U, int e) const {
    // TODO: Task 3.8
    // Compute the stress in element `e` induced by displacement field `U`.
    SymmetricMatrixValue<double, 3> result; // Note: SymmetricMatrixValue constructor zero-initializes!

    return result;
}

TopologyOptimizer::VXd TopologyOptimizer::maxPrincipalStresses(const MX3d &U) const {
    VXd result(numElements());
    for (int e = 0; e < numElements(); ++e)
        result[e] = cauchyStress(U, e).maxMagnitudeEigenvalue();
    return result;
}
