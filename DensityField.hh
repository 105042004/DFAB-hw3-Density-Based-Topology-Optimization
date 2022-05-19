////////////////////////////////////////////////////////////////////////////////
// DensityField.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Representation of a per-elmement density field that can be controlled by
//  various different parametrizations/choices of variables.
//  The changes of variables transforming between these different parametrizations
//  are implemented by `DensityFilter` objects.
//
//  In the simplest parametrization, the variables are simply the per-tet
//  densities. We can change to a smaller set of per-voxel density variables
//  using `VoxelBasedAdapterFilter`. Then we can promote smoother,
//  checkerboard-free designs using the `VoxelSmoothingFilter`.
//  Finally, we can use the `ManufacturingFilter` to ensure a self-supporting
//  design for 3D printing.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
////////////////////////////////////////////////////////////////////////////////
#ifndef DENSITYFIELD_HH
#define DENSITYFIELD_HH
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

// A transformation or change of variables
struct DensityFilter {
    using VXd = Eigen::VectorXd;
    virtual int  inputSize() const = 0;
    virtual int outputSize() const = 0;
    virtual VXd    apply(Eigen::Ref<const VXd>       x) = 0;
    virtual VXd backprop(Eigen::Ref<const VXd> dJ_dout) const = 0;
    virtual ~DensityFilter() { }
};

struct DensityField {
    DensityField(int numElements) { m_rho.setZero(numElements); }
    using VXd = Eigen::VectorXd;
    void setVars(Eigen::Ref<const VXd> x) {
        m_vars = x;
        m_rho = densitiesForVars(x);
    }

    VXd densitiesForVars(Eigen::Ref<const VXd> x) {
        VXd result = x;
        // Apply the filters in order from right-to-left (last to first)
        for (size_t fi = 0; fi < filters.size(); ++fi) {
            const auto &f = filters[filters.size() - 1 - fi];
            if (f->inputSize() > -1 && result.size() != f->inputSize()) throw std::runtime_error("Filter size mismatch");
            result = f->apply(result);
        }
        return result;
    }

    VXd backprop(Eigen::Ref<const VXd> dJ_drho) const {
        VXd result = dJ_drho;
        // Backprop through the filters in reverse order, from left-to-right
        for (auto &f : filters) {
            if (f->outputSize() > -1 && result.size() != f->outputSize()) throw std::runtime_error("Filter size mismatch");
            result = f->backprop(result);
        }
        return result;
    }

    int numVars() {
        // Number of variables is the input size of the first (rightmost) filter
        for (int fi = filters.size() - 1; fi >= 0; --fi) {
            int s = filters[fi]->inputSize();
            if (filters[fi]->inputSize() > -1) // -1 means filter doesn't change size--take size from previous filter...
                return s;
        }
        return m_rho.size();
    }

    const VXd &getVars() const { return m_vars; }
    const VXd &rho() const { return m_rho; }
    double rho(int i) const { return m_rho[i]; }

    std::vector<std::unique_ptr<DensityFilter>> filters;
private:
    VXd m_rho, m_vars;
};

struct ProjectionFilter : public DensityFilter {
    ProjectionFilter() { }

    virtual int  inputSize() const override { return -1; } // Does not change input/output size...
    virtual int outputSize() const override { return -1; }

    virtual VXd apply(Eigen::Ref<const VXd> x) override {
        m_x = x;
        // return 0.5 * (tanh(0.5 * beta) + tanh(beta * (x.array() - 0.5))) / tanh(0.5 * beta);
        return 0.5 * (1.0 + tanh(beta * (x.array() - 0.5)) / tanh(0.5 * beta));
    }

    virtual VXd backprop(Eigen::Ref<const VXd> dJ_dout) const override {
        return 0.5 * beta * dJ_dout.array() / (tanh(0.5 * beta) * pow(cosh(beta * (m_x.array() - 0.5)), 2));
    }

    VXd m_x; // Current value of input variables saved by `apply` for use in `backprop`.

    // Beta > 0 defines the steepness of the Heaviside-like projection 
    // (for beta->inf, projection is the step function)
    float beta = 1.0;
};

////////////////////////////////////////////////////////////////////////////////
// VoxelBasedAdapterFilter
////////////////////////////////////////////////////////////////////////////////
struct VoxelBasedAdapterFilter : public DensityFilter {
    VoxelBasedAdapterFilter(int nv, const Eigen::VectorXi &vfe)
        : numVoxels(nv), voxelForElement(vfe) { }
    virtual int  inputSize() const override { return numVoxels; }
    virtual int outputSize() const override { return voxelForElement.size(); }

    virtual VXd apply(Eigen::Ref<const VXd> x) override {
        const int ne = voxelForElement.size();
        VXd result(ne);
        for (int i = 0; i < ne; ++i)
            result[i] = x[voxelForElement[i]];
        return result;
    }

    virtual VXd backprop(Eigen::Ref<const VXd> dJ_dout) const override {
        VXd result(VXd::Zero(numVoxels));
        const int ne = voxelForElement.size();

        for (int i = 0; i < ne; ++i)
            result[voxelForElement[i]] += dJ_dout[i];

        return result;
    }

    int numVoxels;
    Eigen::VectorXi voxelForElement; // tie each element back to its originating voxel
};


////////////////////////////////////////////////////////////////////////////////
// VoxelSmoothingFilter
////////////////////////////////////////////////////////////////////////////////
struct VoxelSmoothingFilter : public DensityFilter {
    VoxelSmoothingFilter(int numVoxels, Eigen::Vector3i gridShape);

    virtual int  inputSize() const override { return A.cols(); }
    virtual int outputSize() const override { return A.rows(); }

    virtual VXd apply(Eigen::Ref<const VXd> x) override { return A * x; }
    virtual VXd backprop(Eigen::Ref<const VXd> dJ_dout) const override { return A.transpose() * dJ_dout; }

    Eigen::SparseMatrix<double> A;
};

////////////////////////////////////////////////////////////////////////////////
// Self-Supporting Filter
////////////////////////////////////////////////////////////////////////////////
struct VoxelSelfSupportingFilter : public DensityFilter {
    VoxelSelfSupportingFilter(int nv, Eigen::Vector3i gs)
        : numVoxels(nv), gridShape(gs) { }

    virtual int  inputSize() const override { return numVoxels; }
    virtual int outputSize() const override { return numVoxels; }

    virtual VXd apply(Eigen::Ref<const VXd> x) override;
    virtual VXd backprop(Eigen::Ref<const VXd> dJ_dout) const override;

    int numVoxels;
    Eigen::Vector3i gridShape;

    VXd m_input_x;    // Current value of input variables saved by `apply` for use in `backprop`.
    VXd m_filtered_x; // Current value of output variables saved by `apply` for use in `backprop`.

    void getSupportingNeighbors(const Eigen::Vector3i &cell, std::vector<int> &supportingNeighbors) const;

    static double smax(const VXd &x, const std::vector<int> &indices) {
        double sum = 0;
        for (int i : indices) sum += std::pow(x[i], m_P);
        // Exponent used to correct the P-norm overestimation
        double Q = m_P + std::log(indices.size()) / std::log(0.5);
        return std::pow(sum, 1 / Q);
    }

    // Derivative of smax(x[indices])  with respect to x[indices[k]]
    static double dsmax_dxk(const VXd &x, const std::vector<int> &indices, int k) {
        double sum = 0;
        for (int i : indices) sum += std::pow(x[i], m_P);
        double Q = m_P + std::log(indices.size()) / std::log(0.5);
        return (m_P / Q) * std::pow(sum, 1 / Q - 1.0) * std::pow(x[indices[k]], m_P - 1.0);
    }

    static double smin(double x0, double x1) { return 0.5 * (x0 + x1 - std::sqrt((x0-x1)*(x0-x1) + m_epsilon) + std::sqrt(m_epsilon)); }
    static double dsmin_dx0(double x0, double x1) { return 0.5 * (1 - (x0-x1) / std::sqrt((x0-x1)*(x0-x1) + m_epsilon)); }
    static double dsmin_dx1(double x0, double x1) { return dsmin_dx0(x1, x0); }

    // Coefficient used in the approximation of the min function 
    static constexpr double m_epsilon = 1e-4;
    
    // Value defining the P-norm that approximates the max function
    static constexpr double m_P = 40;
};

#endif /* end of include guard: DENSITYFIELD_HH */
