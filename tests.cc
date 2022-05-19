#include <memory>
#include "TopologyOptimizer.hh"
#include "GridGeneration.hh"

Eigen::Vector3i gridShape;

void REQUIRE_EQ(const std::string &name, double a, double b, const double tol = 1e-10) {
    if (std::abs(a - b) > tol * std::abs(b)) {
        double relError = std::abs((a - b) / b);
        std::cout << name << " TEST FAILURE: " << a << " vs " << b << " (relative error " << relError << ")" << std::endl;;
    }
}

void REQUIRE_EQ(const std::string &name, const Eigen::Vector3d &a, const Eigen::Vector3d &b, const double tol = 1e-10) {
    if ((a - b).norm() > tol * b.norm()) {
        double relError = (a - b).norm() / b.norm();
        std::cout << name << " TEST FAILURE: " << a.transpose() << " vs " << b.transpose() << " (relative error " << relError << ")" << std::endl;;
    }
}

using VXd = Eigen::VectorXd;

inline int vox_idx(const Eigen::Vector3i &coords, const Eigen::Vector3i &gridShape) {
    return coords[0] + coords[1] * gridShape[0] + coords[2] * (gridShape[0] * gridShape[1]);
}

Eigen::VectorXd getUnitPerturbation(const VXd &x) {
    Eigen::VectorXd d;
    d.setRandom(x.size());

    // if (d.size() == gridShape.prod()) {
    //     for (int j = 0; j < gridShape[1]; ++j) {
    //         if (j == gridShape[1] - 2) continue;
    //         for (int i = 0; i < gridShape[0]; ++i) {
    //             for (int k = 0; k < gridShape[2]; ++k) {
    //                 d[vox_idx(Eigen::Vector3i(i, j, k), gridShape)] = 0.0;
    //             }
    //         }
    //     }
    //     // d[vox_idx(Eigen::Vector3i(0, gridShape[1] - 2, 0), gridShape)] = 1.0;
    // }

    return d / d.norm();
}

template<class F>
void fd_tests(const std::string &name, F &&f, const VXd &x, const VXd &g, const double fd_eps = 1e-5) {
    for (size_t i = 0; i < 100; ++i) {
        auto d = getUnitPerturbation(x);
        double fd_grad = (f(x + fd_eps * d) - f(x - fd_eps * d)) / (2.0 * fd_eps);
        REQUIRE_EQ("FD Validation " + name, g.dot(d), fd_grad, /* tol = */ 1e-4);
    }
}

int main(int argc, char *argv[]) {
    BoundaryConditions bc("../data/cantilever.obj");

    int gs_x = 15, gs_y = 9, gs_z = 2;
    Eigen::Vector3i gridSamples(gs_x, gs_y, gs_z);
    gridShape = gridSamples.array() - 1;
    Eigen::Vector3d domainShape = Eigen::Vector3d(20, 10, 10);
    // Choose the z dimensions based on the number of z subdivisions to get as-cube-as-possible voxels.
    domainShape[2] = std::sqrt((domainShape[0] / (gs_x - 1)) * (domainShape[1] / (gs_y - 1))) * (gs_z - 1);
    Eigen::MatrixX3d V;
    Eigen::MatrixX4i T;
    Eigen::VectorXi voxelForTet;
    generateGrid(domainShape, gridSamples, V, T, voxelForTet);
    auto topOpt = std::make_unique<TopologyOptimizer>(V, T);

    const int numVoxels = T.rows() / 5;
    
    bc.type[1] = BoundaryConditions::Type::Force;
    bc.data[1] = Eigen::Vector3f(0, -1000, 0);
    topOpt->applyBoundaryConditions(bc);

    auto evalCompliance = [&](const VXd &x) {
        Eigen::MatrixX3d U;
        topOpt->setVars(x);
        return topOpt->solveEquilibriumProblem(U);
    };

    auto evalVol = [&](const VXd &x) {
        topOpt->setVars(x);
        return topOpt->materialVolume();
    };

    auto runTests = [&](const std::string &name) {
        topOpt->setUniformDensities();
        VXd test_x = topOpt->densities.getVars();
        test_x += 1e-2 * getUnitPerturbation(test_x);
        topOpt->setVars(test_x);

        Eigen::MatrixX3d U;
        topOpt->solveEquilibriumProblem(U);

        fd_tests(name + " volume", evalVol, test_x, topOpt->gradVolume());
        topOpt->setVars(test_x);
        fd_tests(name + " compliance", evalCompliance, test_x, topOpt->gradCompliance(U));
    };

    // Derivative tests for raw density variables.
    runTests("tet-based");

    topOpt->densities.filters.push_back(std::make_unique<ProjectionFilter>());
    runTests("projection filter");

    // Derivative tests for voxel-based variables.
    topOpt->densities.filters.push_back(std::make_unique<VoxelBasedAdapterFilter>(numVoxels, voxelForTet));
    runTests("voxel-based");

    // Derivative tests for smoothed variables.
    topOpt->densities.filters.push_back(std::make_unique<VoxelSmoothingFilter>(numVoxels, gridShape));
    runTests("smoothed");

    // Derivative tests for self-supporting-filtered variables.
    topOpt->densities.filters.push_back(std::make_unique<VoxelSelfSupportingFilter>(numVoxels, gridShape));
    runTests("self-supporting");

    // Derivative test for L-BFGS-B wrapper
    {
        SoftVolumeConstrainedProblem prob(*topOpt);
        VXd test_x = topOpt->densities.getVars();
        VXd g;
        prob(test_x, g);
        VXd dummy_g;
        fd_tests("SoftVolumeConstrainedProblem", [&](const VXd &x) { return prob(x, dummy_g); }, test_x, g);
    }

    for (size_t t = 0; t < 100; ++t) {
        std::vector<int> indices{0, 1, 2, 3, 4};
        auto smax_f = [&](const VXd &x) { return VoxelSelfSupportingFilter::smax(x, indices); };

        Eigen::VectorXd x; x.setRandom(5);
        x.array() += 2.0;
        x /= 3.0;

        Eigen::VectorXd g(5);
        for (int k = 0; k < 5; ++k) {
            g[k] = VoxelSelfSupportingFilter::dsmax_dxk(x, indices, k);
        }

        fd_tests("smax", smax_f, x, g);

        x.setRandom(2);
        auto smin_f = [&](const VXd &x) { return VoxelSelfSupportingFilter::smin(x[0], x[1]); };
        g.resize(2);
        g[0] = VoxelSelfSupportingFilter::dsmin_dx0(x[0], x[1]);
        g[1] = VoxelSelfSupportingFilter::dsmin_dx1(x[0], x[1]);
        fd_tests("smin", smin_f, x, g, 1e-6);
    }

    return 0;
}
