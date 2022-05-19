////////////////////////////////////////////////////////////////////////////////
// VisualizationGeometry.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Visualization of all tetrahedra's boundary triangles.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
////////////////////////////////////////////////////////////////////////////////
#ifndef VISUALIZATION_HH
#define VISUALIZATION_HH

template<class KeepPredicate>
void visualizationGeometry(const Eigen::MatrixX3d &V, const Eigen::MatrixX4i &T,
                           double shrinkFactor,
                           Eigen::MatrixX3d &Vvis, Eigen::MatrixX3i &Fvis,
                           const KeepPredicate &keep = [](int i) { return true; }) {
    const int nt = T.rows();

    //      3
    //      *
    //     / \`.
    //    /   \ `* 2
    //   / _.--\ /
    // 0*-------* 1
    Vvis.resize(4 * nt, 3);
    Fvis.resize(4 * nt, 3);

    Eigen::Array<int, 4, 3> tetFaces;
    tetFaces << 0, 2, 1,
                1, 2, 3,
                0, 3, 2,
                0, 1, 3;

    int back = 0;
    for (int i = 0; i < nt; ++i) {
        if (!keep(i)) continue;
        Eigen::RowVector3d c = 0.25 * (V.row(T(i, 0)) + V.row(T(i, 1)) + V.row(T(i, 2)) + V.row(T(i, 3))); // barycenter
        Fvis.block<4, 3>(4 * back, 0) = 4 * back + tetFaces;
        Vvis.block<4, 3>(4 * back, 0) <<
            (1.0 - shrinkFactor) * V.row(T(i, 0)) + shrinkFactor * c,
            (1.0 - shrinkFactor) * V.row(T(i, 1)) + shrinkFactor * c,
            (1.0 - shrinkFactor) * V.row(T(i, 2)) + shrinkFactor * c,
            (1.0 - shrinkFactor) * V.row(T(i, 3)) + shrinkFactor * c;
        ++back;
    }

    Vvis.conservativeResize(4 * back, 3);
    Fvis.conservativeResize(4 * back, 3);
}
#endif /* end of include guard: VISUALIZATION_HH */
