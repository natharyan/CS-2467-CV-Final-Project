#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;

using namespace std; 

struct Observation {
    int point_id;    // Index of 3D point
    int camera_id;   // Index of camera
    Vector2d uv;     // Observed 2D point
};

void bundleAdjustment(
    vector<Vector3d>& points3D,        // 3D points
    vector<Matrix3d>& rotations,       // Camera rotation matrices
    vector<Vector3d>& translations,    // Camera translation vectors
    const vector<Observation>& observations, // Observations
    int iterations = 100, double learning_rate = 1e-4
) {

    for (int iter = 0; iter < iterations; ++iter) {
        // Accumulating gradients for updates
        vector<Vector3d> point_grad(points3D.size(), Vector3d::Zero());
        vector<Matrix3d> rotation_grad(rotations.size(), Matrix3d::Zero());
        vector<Vector3d> translation_grad(translations.size(), Vector3d::Zero());
        
        double total_error = 0.0;

        // Computing gradients
        for (const auto& obs : observations) {
            int p_id = obs.point_id; // Get the point index
            int c_id = obs.camera_id; // Get the camera index

            // Projecting 3D point to 2D using the camera parameters
            Vector3d point_cam = rotations[c_id] * points3D[p_id] + translations[c_id];
            Vector2d proj(point_cam.x() / point_cam.z(), point_cam.y() / point_cam.z());

            // Computing reprojection error
            Vector2d error = proj - obs.uv;
            total_error += error.squaredNorm(); 

            // Computing gradients 
            double z_inv = 1.0 / point_cam.z();
            Vector3d dz_dpoint = -point_cam * z_inv * z_inv;
            Vector3d grad_point = rotations[c_id].transpose() * dz_dpoint;

            point_grad[p_id] += grad_point; 
            translation_grad[c_id] += grad_point; 
        }


        for (size_t i = 0; i < points3D.size(); ++i) {
            points3D[i] -= learning_rate * point_grad[i]; // Update 3D points
        }
        for (size_t i = 0; i < translations.size(); ++i) {
            translations[i] -= learning_rate * translation_grad[i]; // Update translations
        }

        // Logging iteration info
        cout << "Iteration " << iter + 1 << ": Total Error = " << total_error << endl;
    }
}

// int main() {

//     vector<Vector3d> points3D = { {1, 1, 10}, {2, 2, 15}, {3, 3, 20} };
//     vector<Matrix3d> rotations = { Matrix3d::Identity(), Matrix3d::Identity() };
//     vector<Vector3d> translations = { Vector3d::Zero(), Vector3d(1, 1, 1) };

//     vector<Observation> observations = {
//         {0, 0, {0.1, 0.1}}, {1, 0, {0.2, 0.2}}, {2, 1, {0.3, 0.3}}
//     };


//     bundleAdjustment(points3D, rotations, translations, observations);

//     return 0;
// }
