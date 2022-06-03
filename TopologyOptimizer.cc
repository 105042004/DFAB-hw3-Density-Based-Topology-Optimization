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
    Eigen::Matrix3d strain;
    Eigen::Vector3d v1, v2, grad_basis;
    Eigen::Matrix3d I(Eigen::Matrix3d::Identity(3, 3));

    for (int v = 0; v < 4; v++ ) {
        if (v == 0){
            v1 = m_V.row(m_T(e, 3)) - m_V.row(m_T(e, 1));
            v2 = m_V.row(m_T(e, 2)) - m_V.row(m_T(e, 1));
        }
        else if (v == 1) {
            v1 = m_V.row(m_T(e, 2)) - m_V.row(m_T(e, 0));
            v2 = m_V.row(m_T(e, 3)) - m_V.row(m_T(e, 0));
        }
        else if (v == 2) {
            v1 = m_V.row(m_T(e, 3)) - m_V.row(m_T(e, 0));
            v2 = m_V.row(m_T(e, 1)) - m_V.row(m_T(e, 0));
        }
        else if (v == 3) {
            v1 = m_V.row(m_T(e, 1)) - m_V.row(m_T(e, 0));
            v2 = m_V.row(m_T(e, 2)) - m_V.row(m_T(e, 0));
        }
        grad_basis = v1.cross(v2) / (6.0 * m_vols(e));

        for (int d = 0; d < 3; d++ ) {
            strain = 0.5 * (grad_basis * I.col(d).transpose() + I.col(d) * grad_basis.transpose());
            result[v*3 + d] = strain;
        }
    }

    // result[0] = Eigen::Matrix3d::Zero(); // You can compute each strain in an Eigen::Matrix3d despite the compact `SymmetricMatrix` type it's converted to.

    return result;
}

// Compute the *full density* per-element stiffness matrix.
TopologyOptimizer::PerElementStiffnessMatrix TopologyOptimizer::perElementStiffnessMatrix(int e) const {
    PerElementStiffnessMatrix Ke;
    // TODO: Task 3.2
    // Construct the 12x12 per-element stiffness matrix.
    Ke.setIdentity();
    StrainPhis strainVecPhis = getStrainVecPhis(e);

    for (int a = 0; a < 12; a++) {
        auto stress_a = C.doubleContract(strainVecPhis[a]);
        for (int b = a; b < 12; b++) {
            auto stress_ab = stress_a.doubleContract(strainVecPhis[b]);
            Ke(a, b) = stress_ab;
        }
    }

    Ke.triangularView<Eigen::Lower>() = Ke.transpose();
    Ke = Ke * m_vols(e);

    return Ke;
}

TopologyOptimizer::SpMat TopologyOptimizer::buildStiffnessMatrix() const {
    const int numVars = 3 * numVertices();
    SpMat K(numVars, numVars);
    // TODO: Task 3.3
    // Assemble (the lower triangle of) the global stiffness matrix.
    // Remember to apply the SIMP interpolation law!
    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;

    for (int e = 0; e < numElements(); e++) {
        PerElementStiffnessMatrix Ke = (Y_min + (1 - Y_min) * pow(densities.rho(e), p)) * perElementStiffnessMatrix(e);

        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                int rowIndex = 3*m_T(e, i/3) + i%3;
                int colIndex = 3*m_T(e, j/3) + j%3; 

                triplets.emplace_back( rowIndex, colIndex, Ke(i, j));
            }
        }
    }
    K.setFromTriplets(triplets.begin(), triplets.end());

    // K.setIdentity();
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
        for (int i = 0; i < K.outerSize(); ++i) {           // loop over columns
            for (SpMat::InnerIterator it(K, i); it; ++it) { // loop over nonzeros in col

                if (m_isSupportVar[it.col()] || m_isSupportVar[it.row()]) { // current var is constrain vertex, set to I
                    if (it.col() == it.row()) { // on diagonal, set to 1
                        it.valueRef() = 1.0;
                    }
                    else {
                        it.valueRef() = 0.0;
                    }
                }
            }
        }

        // overwrite corresponding entries with zero
        for (int i = 0; i < f.size(); i++) {
            if (m_isSupportVar[i]) {
                f(i) = 0.0;
            }
        }

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

    for (int e = 0; e < numElements(); e++) {
        PerElementStiffnessMatrix Ke = perElementStiffnessMatrix(e);
        VXd u_e(12);
        u_e <<  U.row(m_T(e, 0)).transpose(),
                U.row(m_T(e, 1)).transpose(),
                U.row(m_T(e, 2)).transpose(),
                U.row(m_T(e, 3)).transpose();

        dJ_drho(e) = -p * (1 - Y_min) * pow(densities.rho(e), p-1) * u_e.transpose() * Ke * u_e;
    }

    return densities.backprop(dJ_drho);
}

TopologyOptimizer::VXd TopologyOptimizer::gradVolume() const {
    VXd dV_drho;

    // TODO: Task 3.5
    // Compute the gradient of volume with respect to each tet's density parameter.
    // dV_drho.setZero(numElements());
    dV_drho = m_vols;

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
    MX3d U;

    for(size_t i = 0; i < numSteps; ++i) {

        solveEquilibriumProblem(U);

        auto steppedVarsForLambda = [&](double lambda) -> VXd {
            // Evaluate the element densities rho^new(lambda) corresponding to
            // Lagrange multiplier estimate `lambda`. The formula for this
            // is given in the handout.

            VXd rhos = densities.rho();
            VXd rho_new(numElements());
            
            VXd m_vec(numElements());
            m_vec = m * VXd::Ones(numElements());

            VXd grad_comp = gradCompliance(U);
            VXd grad_vol = gradVolume();

            rho_new = rhos.array() * pow(-grad_comp.array() / (lambda * grad_vol.array()), 0.5);

            rho_new.cwiseMax(0.0).cwiseMax(rhos - m_vec).cwiseMin(1.0).cwiseMin(rhos + m_vec);
            
            return rho_new;
        };

        auto constraint_eval = [&](double lambda) {
            // Evluate the volume constraint violation c(lambda) corresponding
            // to the Lagrange multiplier estimate `lambda`.
            // You will need to use `steppedVarsForLambda(lambda)` to determine
            // the beam areas and then calculate the corresponding volume.
            double result;

            result = maxVolumeFrac * domainVolume() - m_vols.dot(steppedVarsForLambda(lambda));
            
            return result;
        };

        // Bracket the root
        while (constraint_eval(lambda_min) > 0) { lambda_max = lambda_min; lambda_min /= 2; }
        while (constraint_eval(lambda_max) < 0) { lambda_min = lambda_max; lambda_max *= 2; }
        
        // while (constraint_eval(lambda_min) > 0) { lambda_min /= 2; }
        // while (constraint_eval(lambda_max) < 0) { lambda_max *= 2; }

        // Binary Search
        double lambda_mid = 0.5 * (lambda_min + lambda_max);
        double vol_violation = constraint_eval(lambda_mid);

        while (std::abs(vol_violation) > ctol) {
            if (vol_violation < 0) lambda_min = lambda_mid;
            if (vol_violation > 0) lambda_max = lambda_mid;
            lambda_mid = 0.5 * (lambda_min + lambda_max);
            vol_violation = constraint_eval(lambda_mid);
        }

        setVars(steppedVarsForLambda(lambda_mid));
    }

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
