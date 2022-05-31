////////////////////////////////////////////////////////////////////////////////
// Code for ECS 289H Assignment 3
////////////////////////////////////////////////////////////////////////////////
#include "Viewer.hh"
#include <unistd.h>
#include <memory>

#include <igl/writeOBJ.h>
#include <igl/jet.h>
#include <LBFGSB.h>
#include "GridGeneration.hh"
#include "VisualizationGeometry.hh"
#include "TopologyOptimizer.hh"

enum class DensityVariableType : int { PerTet = 0, PerVoxel = 1, Smoothed = 2, Manufacturable = 3 };
enum class OptAlgorithm : int { OC = 0, LBFGSB = 1 };

int main(int argc, const char *argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: " << argv[0] << " region.obj [config.txt]" << std::endl;
        return -1;
    }
    std::string bcPath(argv[1]), configPath;
    if (argc == 3) configPath = argv[2];
    
    BoundaryConditions bc(bcPath);

    // Basic grid generation
    std::unique_ptr<TopologyOptimizer> topOpt;

    Eigen::VectorXi regionForVertex;
    float cachedCompliance = 0.0;
    Eigen::MatrixX3d U;

    int gs_x = 14, gs_y = 8, gs_z = 1;
    Eigen::Vector3f domainShape(20, 10, 10);

    if (!configPath.empty()) {
        std::ifstream inFile(configPath);
        if (!inFile.is_open()) throw std::runtime_error(std::string("Couldn't open input file ") + configPath);
        inFile >> gs_x >> gs_y >> gs_z
               >> bc
               >> domainShape[0] >> domainShape[1] >> domainShape[2];
    };

    auto writeConfig = [&](const std::string &path) {
        std::ofstream outFile(path);
        if (!outFile.is_open()) throw std::runtime_error(std::string("Couldn't open output file ") + path);
        outFile << gs_x << ' ' << gs_y << ' ' << gs_z << ' ' << std::endl
                << bc << std::endl
                << domainShape.transpose() << std::endl;
    };

    DensityVariableType varType = DensityVariableType::PerTet;

    auto regenerateGrid = [&]() {
        // Choose the z dimensions based on the number of z subdivisions to get as-cube-as-possible voxels.
        domainShape[2] = std::sqrt((domainShape[0] / gs_x) * (domainShape[1] / gs_y)) * gs_z;
        Eigen::MatrixX3d V;
        Eigen::MatrixX4i T;
        Eigen::VectorXi voxelForTet;
        Eigen::Vector3i gridSamples(gs_x + 1, gs_y + 1, gs_z + 1);
        generateGrid(domainShape.cast<double>(), gridSamples, V, T, voxelForTet);
        V.col(2).array() -= V.col(2).mean(); // center along the Z axis.

        topOpt = std::make_unique<TopologyOptimizer>(V, T);
        topOpt->applyBoundaryConditions(bc);
        topOpt->configure2DPinConstraints(gs_z);

        const int numVoxels = T.rows() / 5;
        Eigen::Vector3i gridShape = gridSamples.array() - 1;
        topOpt->densities.filters.push_back(std::make_unique<ProjectionFilter>());
        if (int(varType) >= int(DensityVariableType::PerVoxel))
            topOpt->densities.filters.push_back(std::make_unique<VoxelBasedAdapterFilter>(numVoxels, voxelForTet));
        if (int(varType) >= int(DensityVariableType::Smoothed))
            topOpt->densities.filters.push_back(std::make_unique<VoxelSmoothingFilter>(numVoxels, gridShape));
        if (int(varType) >= int(DensityVariableType::Manufacturable))
            topOpt->densities.filters.push_back(std::make_unique<VoxelSelfSupportingFilter>(numVoxels, gridShape));
        topOpt->setUniformDensities();

        regionForVertex = bc.lasso.regionIndicesForPoints(V);
        cachedCompliance = 0;
        U.setZero(V.rows(), 3);
    };
    regenerateGrid();

    auto colorForRegion = [&](int i) {
        Eigen::Vector3d c(0, 0, 0);
        if (i >= 0) igl::jet((i + 1.0) / (bc.lasso.numRegions() + 1.0), c.data()); // use the jet colormap while avoiding the dark margins...
        return c;
    };

    Viewer viewer("289H Homework 3 - Density-Based Topology Optimization");
    auto &vdata = viewer.data();
    viewer.core().background_color << 1.0f, 1.0f, 1.0f, 1.0f;

    bool showPoints = true;
    bool showDeformedStructure = true;

    float shrinkFactor = 0.0;

    enum class ScalarField : int {
        Density = 0, Stress = 1
    };
    ScalarField scalarField = ScalarField::Density;

    float minVisDensity = 0.0f;

    auto updateView = [&]() {
        vdata.clear();
        const auto &V = topOpt->getV();
        Eigen::MatrixX3d VDefo = showDeformedStructure ? (V + U) : V;
        Eigen::MatrixX3d Vvis;
        Eigen::MatrixX3i Fvis;
        const auto &rho = topOpt->densities.rho();
        auto keepTet = [&](int t) { return rho[t] >= minVisDensity; };
        visualizationGeometry(VDefo, topOpt->getT(), shrinkFactor, Vvis, Fvis,
                              keepTet);

        Eigen::VectorXd sf = (scalarField == ScalarField::Density)
                                ? rho
                                : topOpt->maxPrincipalStresses(U);
        // Color with a per-tet scalar field.
        Eigen::MatrixXd Ctet;
        Eigen::MatrixXd C(Fvis.rows(), 3);

        igl::jet(sf, /* normalize = */ scalarField == ScalarField::Stress, Ctet);
        for (int e = 0, back = 0; e < topOpt->numElements(); ++e) {
            if (!keepTet(e)) continue;
            C.block<4, 3>(4 * back++, 0).rowwise() = Ctet.row(e);
        }

        vdata.set_mesh(Vvis, Fvis);
        vdata.set_face_based(true);
        vdata.set_colors(C);

        vdata.show_lines = false;
        static bool centered = false;
        if (!centered) viewer.core().align_camera_center(Vvis, Fvis);
        centered = true;

        if (showPoints) {
            vdata.point_size = 5;
            Eigen::MatrixX3d P(V.rows(), 3);
            C.setZero(V.rows(), 3);

            // Color the grid points according to their matching region
            // If the ground structure is not shown we only show points actually attached to beams.
            int back = 0;
            for (int i = 0; i < regionForVertex.rows(); ++i) {
                if (regionForVertex[i] > -1)
                    C.row(back) = colorForRegion(regionForVertex[i]);
                P.row(back++) = VDefo.row(i);
            }
            vdata.set_points(P.topRows(back), C.topRows(back));
        }
    };
    updateView();

    std::string simFailure;
    auto runSimulation = [&]() {
        simFailure.clear();
        try {
            cachedCompliance = topOpt->solveEquilibriumProblem(U);
        }
        catch (std::exception &e) {
            simFailure = std::string("Simulation failed: ") + e.what();
            U.setZero(topOpt->getV().rows(), 3);
            cachedCompliance = -1;
            std::cerr << simFailure << std::endl;
        }
        updateView();
    };

    int optIterations = 5;
    OptAlgorithm optAlgorithm = OptAlgorithm::OC;

    viewer.menu.callback_draw_viewer_menu = [&]() {
        bool handled = false, bcUpdate = false, forcesUpdated = false, optUpdate = false, visUpdate = false, gridUpdate = false;
        ImGui::Text("Elements: %i, Vertices: %i", topOpt->numElements(), topOpt->numVertices());
        ImGui::Indent();
            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
                gridUpdate |= ImGui::SliderFloat("w", &domainShape[0], /* vmin */ 1.0f, /* vmax */ 60.f, /* format */ "%0.1f"); ImGui::SameLine();
                gridUpdate |= ImGui::SliderFloat("h", &domainShape[1], /* vmin */ 1.0f, /* vmax */ 60.f, /* format */ "%0.1f");
            ImGui::PopItemWidth();
            if (ImGui::InputInt("Grid x size", &gs_x)) { gridUpdate = true; gs_x = std::min<int>(std::max<int>(gs_x, 1), 200); }
            if (ImGui::InputInt("Grid y size", &gs_y)) { gridUpdate = true; gs_y = std::min<int>(std::max<int>(gs_y, 1), 200); }
            if (ImGui::InputInt("Grid z size", &gs_z)) { gridUpdate = true; gs_z = std::min<int>(std::max<int>(gs_z, 1), 40); }
        ImGui::Unindent();

        visUpdate |= ImGui::Checkbox("Show Points", &showPoints);
        visUpdate |= ImGui::Checkbox("Show Deformation", &showDeformedStructure);
        visUpdate |= ImGui::Combo("Scalar Field", (int *)&scalarField, "Density\0Stress\0\0");

        visUpdate |= ImGui::SliderFloat("Tet Shrink Factor", &shrinkFactor, /* vmin */ 0.0f, /* vmax */ 1.0f, /* format */ "%0.6f");
        visUpdate |= ImGui::SliderFloat("Density Vis Cutoff", &minVisDensity, /* vmin */ 0.0f, /* vmax */ 1.0f, /* format */ "%0.6f");

        optUpdate |= ImGui::SliderFloat("Target Volume Frac", &topOpt->maxVolumeFrac, /* vmin */ 0.000001f, /* vmax */ 1.0f, /* format */ "%0.6f");

        optUpdate |= ImGui::SliderFloat("Projection Beta", &dynamic_cast<ProjectionFilter *>(topOpt->densities.filters.front().get())->beta, /* vmin */ 1.0f, /* vmax */ 16.0f, /* format */ "%0.6f");

        // Configuration for each bc region's condition.
        for (int c = 0; c < bc.numConditions(); ++c) {
            auto color = (255 * colorForRegion(c)).cast<unsigned char>().eval();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(ImColor(color[0], color[1], color[2])));
            if (ImGui::Combo(("Condition " + std::to_string(c) + " Type").c_str(), (int *) &bc.type[c], "Support\0Force\0\0")) {
                bc.setType(c, bc.type[c]);
            }
            ImGui::Indent();
            if (bc.type[c] == BoundaryConditions::Type::Force) {
                for (int d = 0; d < 3; ++d)
                    forcesUpdated |= ImGui::SliderFloat((std::string("Force ") + ("xyz"[d]) + "##" + std::to_string(c)).c_str(), &bc.data[c][d], /* vmin */ -50.0f, /* vmax */ 50.0f, /* format */ "%0.6f");
            }
            if (bc.type[c] == BoundaryConditions::Type::Support) {
                for (int d = 0; d < 3; ++d) {
                    forcesUpdated |= ImGui::Checkbox(("xyz"[d] + std::string("##support") + std::to_string(c)).c_str(), (bool *)&bc.data[c][d]);
                    if (d < 2) ImGui::SameLine();
                }
            }
            ImGui::Unindent();
            ImGui::PopStyleColor(1);
        }

        if (ImGui::Button("Save Config")) {
            try {
                writeConfig(igl::file_dialog_save());
            }
            catch (...) { }
            handled = true;
        }
        ImGui::SameLine();

        if (ImGui::Button("Save Grid")) {
            try {
                auto path = igl::file_dialog_save();
                igl::writeOBJ(path, topOpt->getV(), topOpt->getBdryF());
            }
            catch (...) { }
            handled = true;
        }

        ImGui::Text("Compliance: %f", cachedCompliance);

        gridUpdate |= ImGui::Combo("Var Type", (int *) &varType, "Per Tet\0Per Voxel\0Smoothed\0Manufacturable\0\0");

        if (gridUpdate) {
            regenerateGrid();
            bcUpdate = true;
            visUpdate = true;
        }

        handled |= ImGui::Combo("Algorithm", (int *) &optAlgorithm, "OC\0LBFGS-B\0\0");

        if (ImGui::InputInt("Iterations", &optIterations)) {
            optIterations = std::min(std::max(optIterations, 1), 80);
            handled = true;
        }

        if (ImGui::Button("Simulate")) {
            runSimulation();
            return true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Optimize")) {
            if (optAlgorithm == OptAlgorithm::OC)
                topOpt->optimizeOC(optIterations);
            if (optAlgorithm == OptAlgorithm::LBFGSB) {
                LBFGSpp::LBFGSBParam<double> param;
                // Work around LBFGSpp's weird convergence test;
                // Its relative gradient tolerance is relative to the norm of `x` instead of
                // the initial gradient...
                param.epsilon_rel = 0.0;
                param.epsilon = 1e-6;
                param.max_iterations = optIterations;
                auto x = topOpt->densities.getVars();
                LBFGSpp::LBFGSBSolver<double> solver(param);
                double opt_f;
                SoftVolumeConstrainedProblem prob(*topOpt);
                try {
                     solver.minimize(prob, x, opt_f, /* lb = */ Eigen::VectorXd::Zero(x.size()), /* ub = */ Eigen::VectorXd::Ones(x.size()));
                }
                catch (const std::exception &e) {
                    std::cout << "LBFGS exception: " << e.what() << std::endl;
                }
                
                topOpt->setVars(x);
            }

            runSimulation();
            return true;
        }

        if (ImGui::Button("Set Uniform Densities")) {
            topOpt->setUniformDensities();
            runSimulation();
            return true;
        }

        if (ImGui::Button("Dump Densities")) {
            std::ofstream varsFile("vars.txt");
            varsFile << topOpt->densities.getVars() << std::endl;

            std::ofstream dfile("densities.txt");
            dfile << topOpt->densities.rho() << std::endl;
        }

        if (bcUpdate || forcesUpdated) {
            topOpt->applyBoundaryConditions(bc);
            topOpt->configure2DPinConstraints(gs_z);
        }

        if (forcesUpdated)
            runSimulation();

        if (visUpdate)
            updateView();

        return handled || bcUpdate || optUpdate || forcesUpdated;
    };

    viewer.run();

    return 0;
}
