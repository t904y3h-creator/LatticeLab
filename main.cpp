#include "App/Application.h"

int main() {
    Application application;
    return application.run();
}

// #include <fstream>

// #include "App/AppActions.h"
// #include "App/CreateWindow.h"
// #include "App/Scenes.h"
// #include "Engine/Simulation.h"
// #include "Engine/gpu/WGPUContext.h"

// int main() {
//     GLFWwindow* window = createWindow();
//     if (!window) {
//         return EXIT_FAILURE;
//     }

//     int width, height;
//     glfwGetFramebufferSize(window, &width, &height);
//     WGPUContext::instance().init(window, width, height);

//     // инициализация систем
//     SimBox box(Vec3f(50, 50, 6));
//     Simulation simulation(box);
//     // создание сцены
//     Scenes::crystal(simulation, 50, AtomData::Type::Z, false);

//     simulation.enableGpu(true);
//     for (size_t i = 0; i < 1000; ++i) {
//         simulation.update();
//     }
//     auto save_binary = [](std::string_view filename, std::span<const float> data) {
//         std::ofstream out(filename.data(), std::ios::binary);
//         if (out.is_open()) {
//             out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
//         }
//         out.close();
//     };
//     save_binary("gpu_verlet_correct_1000.tmp", simulation.atoms().floatDataSpan());

//     return 0;
// }
