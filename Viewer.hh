////////////////////////////////////////////////////////////////////////////////
// Viewer.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Simple wrapper around libigl's viewer that sets our preferred defaults.
*/
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
////////////////////////////////////////////////////////////////////////////////
#ifndef VIEWER_HH
#define VIEWER_HH

#include <igl/opengl/glfw/Viewer.h>
#include <GLFW/glfw3.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

using IGLViewer = igl::opengl::glfw::Viewer;

struct Viewer : public IGLViewer {
    Viewer(const std::string &title = "289H Viewer")
        : windowTitle(title)
    {
        data().set_face_based(true);
        core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);

        plugins.push_back(&plugin);
        plugin.widgets.push_back(&menu);

        launch_init(/* resizeable = */ true,
                    /* fullscreen = */ false,
                    /*       name = */ windowTitle);

        // Redraw when the window is resizing
        // This must be set after `launch_init` otherwise we'll try to redraw
        // before the menu plugin is initialized and crash.
        bool reentered_from_draw = false;
        IGLViewer::callback_post_resize = [this, reentered_from_draw](IGLViewer &v, int w, int h) mutable {
            if (Viewer::callback_post_resize)
                return Viewer::callback_post_resize(v, w, h);
            if (!reentered_from_draw) {
                reentered_from_draw = true;
                v.draw();
                glfwSwapBuffers(v.window);
                reentered_from_draw = false;
            }
            return true;
        };

        // Customize the ImGui menu window
        menu.callback_draw_viewer_window = [&]() {
          float menu_width = 200.f * menu.menu_scaling();
          ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
          ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
          ImGui::SetNextWindowSizeConstraints(ImVec2(menu_width, -1.0f), ImVec2(menu_width, -1.0f));
          bool _viewer_menu_visible = true;
          ImGui::Begin(
              "Settings", &_viewer_menu_visible,
              ImGuiWindowFlags_NoSavedSettings
              | ImGuiWindowFlags_AlwaysAutoResize
          );
          ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.4f);
          if (menu.callback_draw_viewer_menu) { menu.callback_draw_viewer_menu(); }
          else { menu.draw_viewer_menu(); }
          ImGui::PopItemWidth();
          ImGui::End();
        };
    }

    // The libigl viewer specifies its current camera as a transformation of a
    // "base camera," which is chosen to approximately fit the mesh
    // data in view.
    // Normally they configure the base camera at `launch_init`, but this won't happen
    // for us since we call this before any mesh data is added. We expose this
    // convenience method for the user to reinitialize the base camera after
    // setting data.
    void recalculate_base_camera() { core().align_camera_center(data().V, data().F); }

    // We also provide a convenience method that allows automatically updating
    // the base camera when setting new mesh data.
    void set_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, bool update_base_camera = false) {
        data().set_mesh(V, F);
        if (update_base_camera) recalculate_base_camera();
    }

    bool run() { return launch_rendering(/* loop = */ true); }

    // Still allow the user to specify their own custom post-resize callback
    // (in addition to our default one that enables redraw during resize).
    std::function<bool(IGLViewer &viewer, int w, int h)> callback_post_resize;

    std::string windowTitle;

    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    ~Viewer() {
        launch_shut();
    }
};

#endif /* end of include guard: VIEWER_HH */
