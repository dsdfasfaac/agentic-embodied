### Policy interface and timing

   Setting                               Value
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Action/state dimension                14: left 6 joints + gripper, right 6 joints + gripper
  ────────────────────────────────────  ─────────────────────────────────────────────────────────────────────
   Action semantics                      Absolute joint positions, not deltas or torques
  ────────────────────────────────────  ─────────────────────────────────────────────────────────────────────
   Active robot                          Right arm only: 6 hinges + 2 finger slides
  ────────────────────────────────────  ─────────────────────────────────────────────────────────────────────
   Left channels                         No simulated joints; fixed task-state placeholders
  ────────────────────────────────────  ─────────────────────────────────────────────────────────────────────
   State sent to VLA                     Left seven channels zeroed; right channels measured
  ────────────────────────────────────  ─────────────────────────────────────────────────────────────────────
   Prediction horizon                    32 actions
  ────────────────────────────────────  ─────────────────────────────────────────────────────────────────────
   Actions executed before replanning    First 16
  ────────────────────────────────────  ─────────────────────────────────────────────────────────────────────
   Policy/action rate                    15 Hz
  ────────────────────────────────────  ─────────────────────────────────────────────────────────────────────
   Interpolated command rate             60 Hz
  ────────────────────────────────────  ─────────────────────────────────────────────────────────────────────
   Physics rate during these rollouts    600 Hz, 10 physics steps per command
  ────────────────────────────────────  ─────────────────────────────────────────────────────────────────────
   Batch limit                           32 chunks → up to 512 actions, approximately 34.1 simulated seconds
  ────────────────────────────────────  ─────────────────────────────────────────────────────────────────────
   Environment limit                     600 actions

  The XML specifies a 960 Hz physics rate, but this scene’s runtime overrides it to 600 Hz. Saved trajectory timestamps confirm one action every 0.066667 seconds.

  ### Action filtering and interpolation

  For each active channel, the target is filtered relative to the previous commanded target, not the measured joint position:

  filtered change = 0.35 × (VLA target − previous command)

   Setting                                              Arm               Gripper
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━
   Filter coefficient                                  0.35                  0.35
  ─────────────────────────────────────────────  ───────────  ────────────────────
   Maximum change per policy step                 0.035 rad    0.08 encoder units
  ─────────────────────────────────────────────  ───────────  ────────────────────
   Additional command-jump rejection threshold      0.5 rad     1.0 encoder units

  The current interpolation implementation uses a quintic smoothstep for arm commands across four command updates. Gripper targets are applied directly at each update.

  Important: the task file contains gripper_command_offsets: [0.0, 0.9], but the current control path does not apply that field. It should not be interpreted as an effective +0.9 offset.

  ### Joint and actuator configuration

   Setting                                  Right arm joints 1–6             Each finger joint
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Joint type                                              Hinge                         Slide
  ───────────────────────────────────────  ──────────────────────  ────────────────────────────
   Joint limits                                   −10 to +10 rad    0 to approximately 0.049 m
  ───────────────────────────────────────  ──────────────────────  ────────────────────────────
   Position actuator gain kp                                3000                          4000
  ───────────────────────────────────────  ──────────────────────  ────────────────────────────
   Actuator velocity damping kv                               70                             6
  ───────────────────────────────────────  ──────────────────────  ────────────────────────────
   Passive joint damping                                     0.2                             1
  ───────────────────────────────────────  ──────────────────────  ────────────────────────────
   Armature                                                 0.02                         0.005
  ───────────────────────────────────────  ──────────────────────  ────────────────────────────
   Joint friction loss                                         0                             0
  ───────────────────────────────────────  ──────────────────────  ────────────────────────────
   Actuator force/torque limit                          ±100 N·m                          ±4 N
  ───────────────────────────────────────  ──────────────────────  ────────────────────────────
   Explicit actuator control-range limit                    None                          None
  ───────────────────────────────────────  ──────────────────────  ────────────────────────────
   Gravity compensation                                  Enabled                       Enabled

  Joint limits are solver constraints; actuator force limits are separate. These broad ±10-radian arm limits are simulation settings, not a claim about hardware travel.

  Both fingers receive the same target. There is no equality constraint coupling them, so their actual positions can differ under contact.

  ### Gripper mapping

  The policy uses one scalar (g):

  finger_target_m = −g × 0.044 / 3.4

   Policy value    Meaning    Each finger’s target
  ━━━━━━━━━━━━━━  ━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━
   −3.4            Open                      44 mm
  ──────────────  ─────────  ──────────────────────
   0               Closed                     0 mm

  The commanded range is therefore 0–44 mm, despite the approximately 49 mm physical slide limits. These are individual joint displacements, not directly the jaw aperture.

  Readback averages the two finger positions, clips to the representable 0–44 mm interval, then converts back to policy units. Command conversion rejects values beyond its configured tolerance.

  ### Mass and inertia

  These are the compiled values, including the reconstructed finger assembly. Inertias below are principal moments in each body’s inertial frame, in kg·m²; reproducing them also requires the XML’s inertial position and quaternion.

   Body             Mass, kg    Principal moments of inertia
  ━━━━━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Right link 1     0.066982    (0.000030, 0.000070, 0.000090)
  ──────────────  ───────────  ─────────────────────────────────────────
   Right link 2     1.079500    (0.000509891, 0.015990006, 0.016050103)
  ──────────────  ───────────  ─────────────────────────────────────────
   Right link 3     0.545340    (0.000320578, 0.004221002, 0.004248420)
  ──────────────  ───────────  ─────────────────────────────────────────
   Right link 4     0.117140    (0.000071498, 0.000212860, 0.000275643)
  ──────────────  ───────────  ─────────────────────────────────────────
   Right link 5     0.634880    (0.000251529, 0.000820000, 0.000838471)
  ──────────────  ───────────  ─────────────────────────────────────────
   Right link 6     0.440890    (0.000280, 0.000380, 0.000500)
  ──────────────  ───────────  ─────────────────────────────────────────
   Each finger     0.1166364    (0.000036, 0.000054, 0.000054)

  Total right-robot subtree mass: 3.1180048 kg.

  The compiler uses inertiafromgeom="false". Explicit inertias govern dynamics; decorative meshes do not acquire inferred mass. This setting is essential when transplanting the robot.

  ### Contacts and solver

   Setting                                      Value
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Gravity                                      (0, 0, −9.81) m/s²
  ───────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   Integrator                                   implicitfast
  ───────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   Solver                                       Newton
  ───────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   Solver iterations / tolerance                80 / 1e−9
  ───────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   Friction cone                                Elliptic
  ───────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   Active robot collisions                      Gripper contact geometry; upper-arm/self collisions excluded
  ───────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   Gripper contact dimension                    condim=4
  ───────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   Gripper sliding friction                     0.8 or 1.2, depending on contact geom
  ───────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   Torsional / rolling friction coefficients    0.002 / 0.0001
  ───────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   Gripper contact solref                       (0.006, 1)

  Object-side contact parameters also affect the resulting grasp; copying robot parameters alone does not reproduce contact behavior.

  ### Initial state and observations

  Right-arm initial angles, in radians:

  [0.01354218, −0.00553131, 0.00247956,
   −0.03414250, −0.00057221, 0.01888286]

  Initial gripper policy value: −3.322798, corresponding to approximately 43.0009 mm per finger.

  Reset restores the saved initial qpos and qvel, then initializes robot positions and actuator targets from the task manifest. It does not zero all saved velocities or run an additional settling phase.