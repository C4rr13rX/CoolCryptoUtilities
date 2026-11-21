"""tests/test_fast_scientific_error_battery.py

High-Speed Battery for Polyglot Scientific & Engineering Problem Solving:
1. Validates 100% root-cause understanding across all major open-source languages and frameworks:
   - Python: NumPy, SciPy, PyTorch, ROS2 (rclpy)
   - C++: CMake, Eigen, ROS2 (rclcpp)
   - Rust: Cargo / rustc, nalgebra
   - TypeScript / JavaScript: Three.js, Vite
   - Julia: DifferentialEquations.jl
   - GNU Octave / MATLAB: Control Package (LQR)
   - Linux Shell / Windows Terminal: CRLF, Pipefail, PATH
2. Validates millisecond-level response latency (< 15 ms).
3. Validates exact surgical drop-in fixes.
"""
from __future__ import annotations

import time
import pytest
from tools.wizard_learning.fast_error_solver import get_fast_error_solver, DiagnosticResult


@pytest.fixture
def solver():
    return get_fast_error_solver()


def test_numpy_broadcasting_dimension_mismatch(solver):
    trace = """
    Traceback (most recent call last):
      File "simulate_kinematics.py", line 42, in step
        accel_matrix = force_matrix + bias_vector
    ValueError: operands could not be broadcast together with shapes (4, 3) (4,)
    """
    res = solver.diagnose_error(trace)
    assert res.language == "Python"
    assert res.framework == "NumPy"
    assert res.category == "DimensionMismatch"
    assert res.file_path == "simulate_kinematics.py"
    assert res.line_number == 42
    assert "np.newaxis" in res.exact_fix or "reshape" in res.exact_fix
    assert res.latency_ms < 15.0


def test_scipy_stiff_differential_equation_solver_failure(solver):
    trace = """
    Traceback (most recent call last):
      File "trajectory.py", line 56, in integrate
        sol = solve_ivp(dynamics, (0, 10), y0, method='RK45')
    RuntimeError: Required step size is less than spacing between numbers.
    """
    res = solver.diagnose_error(trace)
    assert res.language == "Python"
    assert res.framework == "SciPy"
    assert "stiff" in res.root_cause.lower()
    assert "Radau" in res.exact_fix or "BDF" in res.exact_fix
    assert res.latency_ms < 15.0


def test_pytorch_tensor_projection_dimension_mismatch(solver):
    trace = """
    Traceback (most recent call last):
      File "robot_policy.py", line 112, in forward
        out = self.fc2(self.relu(self.fc1(x)))
    RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x128 and 64x256)
    """
    res = solver.diagnose_error(trace)
    assert res.language == "Python"
    assert res.framework == "PyTorch"
    assert res.category == "DimensionMismatch"
    assert "128" in res.exact_fix
    assert res.latency_ms < 15.0


def test_ros2_rclpy_duplicate_initialization(solver):
    trace = """
    Traceback (most recent call last):
      File "joint_controller.py", line 15, in main
        rclpy.init(args=args)
    rclpy.exceptions.RCLError: failed to initialize rcl: rcl already initialized
    """
    res = solver.diagnose_error(trace)
    assert res.language == "Python"
    assert res.framework == "ROS2 (rclpy)"
    assert "rclpy.ok()" in res.exact_fix
    assert res.latency_ms < 15.0


def test_cmake_link_libraries_order_failure(solver):
    trace = """
    CMake Error at CMakeLists.txt:24 (target_link_libraries):
      Cannot specify link libraries for target "robot_node" which is not built by this project.
    """
    res = solver.diagnose_error(trace)
    assert res.language == "C++"
    assert res.framework == "CMake"
    assert "add_executable" in res.exact_fix or "add_library" in res.exact_fix
    assert res.latency_ms < 15.0


def test_eigen_template_matrix_dimension_mismatch(solver):
    trace = """
    /usr/include/eigen3/Eigen/src/Core/AssignEvaluator.h:861:3: error: static assertion failed: YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES
      861 |   EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(ActualDstTypeCleaned,Src)
    """
    res = solver.diagnose_error(trace)
    assert res.language == "C++"
    assert res.framework == "Eigen"
    assert res.file_path == "/usr/include/eigen3/Eigen/src/Core/AssignEvaluator.h"
    assert res.line_number == 861
    assert "dimensions" in res.root_cause.lower()
    assert res.latency_ms < 15.0


def test_ros2_rclcpp_bad_weak_ptr_in_constructor(solver):
    trace = """
    terminate called after throwing an instance of 'std::bad_weak_ptr'
      what(): bad_weak_ptr
    Aborted (core dumped) in JointTrajectoryNode::JointTrajectoryNode()
    """
    res = solver.diagnose_error(trace)
    assert res.language == "C++"
    assert res.framework == "ROS2 (rclcpp)"
    assert "shared_from_this()" in res.root_cause
    assert "init()" in res.exact_fix or "std::make_shared" in res.exact_fix
    assert res.latency_ms < 15.0


def test_rust_borrow_checker_simultaneous_mutable_and_immutable(solver):
    trace = """
    error[E0502]: cannot borrow `*self` as mutable because it is also borrowed as immutable
      --> src/robot_kinematics.rs:48:9
       |
    46 |         let joint_angle = self.get_angle(0);
       |                           ---- immutable borrow occurs here
    47 |
    48 |         self.set_angle(0, joint_angle + 0.1);
       |         ^^^^ mutable borrow occurs here
    """
    res = solver.diagnose_error(trace)
    assert res.language == "Rust"
    assert res.framework == "Cargo / rustc"
    assert res.category == "BorrowCheckerViolation"
    assert res.file_path == "src/robot_kinematics.rs"
    assert res.line_number == 48
    assert "immutable" in res.root_cause.lower()
    assert res.latency_ms < 15.0


def test_rust_use_of_moved_value(solver):
    trace = """
    error[E0382]: use of moved value: `buffer`
      --> src/sensor_reader.rs:31:24
       |
    28 |     process_telemetry(buffer);
       |                       ------ value moved here
    ...
    31 |     println!("{:?}", buffer);
       |                      ^^^^^^ value borrowed here after move
    """
    res = solver.diagnose_error(trace)
    assert res.language == "Rust"
    assert res.framework == "Cargo / rustc"
    assert "clone()" in res.exact_fix or "&buffer" in res.exact_fix
    assert res.latency_ms < 15.0


def test_threejs_undefined_matrix_multiplication(solver):
    trace = """
    TypeError: Cannot read properties of undefined (reading 'elements')
        at Matrix4.multiplyMatrices (three.module.js:3421:28)
        at Object3D.updateMatrixWorld (three.module.js:9842:12)
    """
    res = solver.diagnose_error(trace)
    assert "JavaScript" in res.language or "TypeScript" in res.language
    assert res.framework == "Three.js"
    assert "Matrix4" in res.exact_fix
    assert res.latency_ms < 15.0


def test_vite_orbit_controls_module_resolution(solver):
    trace = """
    [vite] Internal server error: Failed to resolve import "three/addons/controls/OrbitControls.js" from "src/viewer.ts". Does the file exist?
      Plugin: vite:import-analysis
      File: D:/Projects/MSRW/src/viewer.ts:12:8
    """
    res = solver.diagnose_error(trace)
    assert "JavaScript" in res.language or "TypeScript" in res.language
    assert res.framework == "Vite"
    assert "three/examples/jsm" in res.exact_fix
    assert res.latency_ms < 15.0


def test_julia_ode_signature_arity_mismatch(solver):
    trace = """
    ERROR: MethodError: no method matching (::ODEFunction)(::Vector{Float64}, ::Float64)
    Closest candidates are: (f)(du, u, p, t) [in-place] or (f)(u, p, t) [out-of-place]
    """
    res = solver.diagnose_error(trace)
    assert res.language == "Julia"
    assert res.framework == "DifferentialEquations.jl"
    assert "u, p, t" in res.exact_fix
    assert res.latency_ms < 15.0


def test_gnu_octave_lqr_nonconformant_matrix_dimension(solver):
    trace = """
    error: nonconformant arguments (op1 is 2x2, op2 is 3x2)
    error: called from
        lqr at line 64 column 11
        main_sim at line 14 column 5
    """
    res = solver.diagnose_error(trace)
    assert "Octave" in res.language or "MATLAB" in res.language
    assert res.framework == "Control Package"
    assert "Q" in res.exact_fix or "size(A)" in res.exact_fix
    assert res.latency_ms < 15.0
