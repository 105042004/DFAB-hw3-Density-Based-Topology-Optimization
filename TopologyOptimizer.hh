////////////////////////////////////////////////////////////////////////////////
// TopologyOptimizer.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Density-based topology optimization.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
////////////////////////////////////////////////////////////////////////////////
#ifndef TOPOLOGYOPTIMIZER_HH
#define TOPOLOGYOPTIMIZER_HH

#include <Eigen/Sparse>
#include "RegionLasso.hh"
#include "ElasticityTensor.hh"
#include "DensityField.hh"

#if HAS_CHOLMOD
#include <Eigen/CholmodSupport>
#endif

struct BoundaryConditions {
    enum class Type : int { Support = 0, Force = 1 };

    BoundaryConditions(const std::string &lassoPath)
        : lasso(lassoPath) {
        type.resize(numConditions(), Type::Support);
        data.resize(numConditions(), Eigen::Vector3f::Zero());
        for (int c = 0; c < numConditions(); ++c)
            setType(c, Type::Support);
    }

    std::vector<Type> type;
    std::vector<Eigen::Vector3f> data; // force vector; reinterpreted as boolean array in support case.

    void setType(int c, Type t) {
        type[c] = t;
        if (t == Type::Support)
            componentSupported(c, 0) = componentSupported(c, 1) = componentSupported(c, 2) = true;
        if (t == Type::Force)
            data[c].setZero();
    }

    bool &componentSupported(int c, int d) { return *((bool*)&data[c][d]); }
    const bool componentSupported(int c, int d) const { return *((bool*)&data[c][d]); }

    int numConditions() const { return lasso.numRegions(); }

    friend std::ostream &operator<<(std::ostream &os, const BoundaryConditions &bc) {
        os << bc.numConditions() << std::endl;
        for (size_t i = 0; i < bc.numConditions(); ++i) {
            if (bc.type[i] == Type::Support)
                os << int(bc.type[i]) << ' ' << bc.componentSupported(i, 0) << ' ' << bc.componentSupported(i, 1) << ' ' << bc.componentSupported(i, 2) << std::endl;
            else
                os << int(bc.type[i]) << ' ' << bc.data[i].transpose() << std::endl;
        }
        return os;
    }

    friend std::istream &operator>>(std::istream &is, BoundaryConditions &bc) {
        int nc;
        is >> nc;
        if (nc != bc.numConditions()) throw std::runtime_error("Condition count mismatch");
        for (size_t i = 0; i < nc; ++i) {
            is >> *((int *)(&bc.type[i]));
            if (bc.type[i] == Type::Support)
                is >> bc.componentSupported(i, 0) >> bc.componentSupported(i, 1) >> bc.componentSupported(i, 2);
            else
                is >> bc.data[i][0] >> bc.data[i][1] >> bc.data[i][2];
        }
        return is;
    }

    RegionLasso lasso;
};

struct TopologyOptimizer {
    using SpMat   = Eigen::SparseMatrix<double>;
    using Triplet = Eigen::Triplet<double>;
    using VXd  = Eigen::VectorXd;
    using VXi  = Eigen::VectorXi;
    using MX3d = Eigen::MatrixX3d;
    using MX3i = Eigen::MatrixX3i;
    using MX4i = Eigen::MatrixX4i;
    using AXb  = Eigen::Array<bool, Eigen::Dynamic, 1>;

    using PerElementStiffnessMatrix = Eigen::Matrix<double, 12, 12>;
    using StrainPhis = std::array<SymmetricMatrixValue<double, 3>, 12>;

    TopologyOptimizer(Eigen::Ref<const MX3d> V, Eigen::Ref<const MX4i> T);

    int numElements() const { return m_T.rows(); }
    int numVertices() const { return m_V.rows(); }

    const MX3d &getV()     const { return m_V;     }
    const MX4i &getT()     const { return m_T;     }
    const MX3i &getBdryF() const { return m_bdryF; }

    StrainPhis getStrainVecPhis(int e) const;

    PerElementStiffnessMatrix perElementStiffnessMatrix(int e) const;
    SpMat buildStiffnessMatrix() const;

    // Solve for the equilibrium displacements `U` of the current design under
    // external forces `m_Fext` and support variables `m_isSupportVar == true`.
    // The compliance of the deformed structure is returned.
    double solveEquilibriumProblem(MX3d &U) const;

    // Compute optimal densities in `densities.rho()` using the optimality criterion method.
    void optimizeOC(int numSteps);

    VXd gradCompliance(const MX3d &U) const;
    VXd gradVolume() const;
    double materialVolume() const { return m_vols.dot(densities.rho()); }
    double domainVolume() const { return m_vols.sum(); }

    void applyBoundaryConditions(const BoundaryConditions &bc) {
        auto regionForVtx = bc.lasso.regionIndicesForPoints(m_V);
        AXb newIsSupportVar;
        newIsSupportVar.setZero(3 * m_V.rows());
        m_Fext.setZero(m_V.rows(), 3);

        // Apply support conditions to all vertices falling in support regions.
        for (int i = 0; i < m_V.rows(); ++i) {
            int r = regionForVtx[i];
            if (r < 0) continue;
            if (bc.type[r] == BoundaryConditions::Type::Support) {
                for (int d = 0; d < 3; ++d)
                    newIsSupportVar[3 * i + d] = bc.componentSupported(r, d);
            }
        }

        // Changing the support nodes invalidates the factorization.
        if (m_isSupportVar.size() == 0 || (newIsSupportVar != m_isSupportVar).any()) {
            m_KFactorizationCache = false;
            m_isSupportVar = newIsSupportVar;
        }

        // Determine the vertex barycentric region area (we're using a lumped
        // mass matrix to compute the load for per-vertex traction fields).
        std::vector<double> barycentricArea(m_V.rows());
        for (int i = 0; i < m_bdryF.rows(); ++i) {
            barycentricArea[m_bdryF(i, 0)] += m_bdryAreas[i] / 3;
            barycentricArea[m_bdryF(i, 1)] += m_bdryAreas[i] / 3;
            barycentricArea[m_bdryF(i, 2)] += m_bdryAreas[i] / 3;
        }

        // Determine the total surface area in each region (for spreading the
        // total region force over this area)
        std::vector<double> regionArea(bc.lasso.numRegions());
        for (int i = 0; i < m_V.rows(); ++i) {
            const int r = regionForVtx[i];
            if (r >= 0) regionArea[r] += barycentricArea[i];
        }

        // Get the force acting on each vertex due to the boundary forces.
        for (int i = 0; i < m_V.rows(); ++i) {
            int r = regionForVtx[i];
            if ((r < 0) || (bc.type[r] != BoundaryConditions::Type::Force)) continue;
            if (barycentricArea[i] <= 0) continue;
            m_Fext.row(i) += barycentricArea[i] / regionArea[r] * bc.data[r].cast<double>();
        }
    }

    // Flatten an X by d matrix `F` in to a vector of length d * X.
    // Uses the x0, y0, z0, x1, ... ordering that is compatible with our `C`
    // matrix (i.e. placing the first row of F in the top `d` entries of the
    // output, followed by the subsequent rows).
    static VXd flatten(Eigen::Ref<const Eigen::MatrixXd> F) {
        Eigen::VectorXd f(F.cols() * F.rows());
        // Note: we use a row-major mapping of the output vector so that row entries are flattened continuous.
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(f.data(), F.rows(), F.cols()) = F;
        return f;
    }

    // Unflatten a vector of length d * X into an X by d matrix
    static void unflatten(Eigen::Ref<const VXd> f, int d, Eigen::MatrixX3d &result) {
        const int rows = f.size() / d;
        if (f.size() % d != 0) throw std::runtime_error("Flattened vector length must be divisible by d");
        if ((result.rows() != rows) || (result.cols() != d))
            result.resize(rows, d);
        result = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(f.data(), rows, d);
    }

    SymmetricMatrixValue<double, 3> cauchyStress(const MX3d &U, int e) const;
    VXd maxPrincipalStresses(const MX3d &U) const;

    void setUniformDensities() {
        // Note: this may only be approximate when density filters are in use!
        setVars(maxVolumeFrac * VXd::Ones(densities.numVars()));
    }

    void setVars(const VXd &x) {
        densities.setVars(x);
        m_KFactorizationCache = false;
    }

#if HAS_CHOLMOD
    using Solver = Eigen::CholmodSupernodalLLT<SpMat>;
#else
    using Solver = Eigen::SimplicialLLT<SpMat>;
#endif
    mutable bool m_KFactorizationCache = false;
    mutable std::unique_ptr<Solver> solver;

    ElasticityTensor<double, 3> C;
    DensityField densities;

    float maxVolumeFrac = 0.5;

    // SIMP exponent
    double p = 3.0;
    double Y_min = 1e-6; // The Young's modulus scale factor to assign void material (nonzero to avoid singular K).
private:
    // Boundary conditions
    MX3d m_Fext; // traction (force per unit area) on each boundary element.
    AXb m_isSupportVar;

    MX3d m_V;     // Rest positions of the structures' vertices.
    MX4i m_T;     // Tetrahedra of the structure
    MX3i m_bdryF; // Faces of the boundary mesh

    VXd m_vols;      // Element volumes
    VXd m_bdryAreas; // Area of each boundary face
};

// LBFGS-compatible soft-volume-constrained formulation.
struct SoftVolumeConstrainedProblem {
    SoftVolumeConstrainedProblem(TopologyOptimizer &to)
        : topOpt(to) { }

    double volPenalty = 100.0;
    double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g) {
        topOpt.setVars(x);

        double volDeviation = (topOpt.materialVolume() - topOpt.maxVolumeFrac * topOpt.domainVolume());
        double J = topOpt.solveEquilibriumProblem(U) + 0.5 * volPenalty * volDeviation * volDeviation;
        g = topOpt.gradCompliance(U) + volPenalty * volDeviation * topOpt.gradVolume();
        return J;
    }

    TopologyOptimizer::MX3d U;
    TopologyOptimizer &topOpt;
};

#endif /* end of include guard: TOPOLOGYOPTIMIZER_HH */
