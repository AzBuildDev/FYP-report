@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

call "%SCRIPT_DIR%视觉对齐与手动修正.bat" ^
  --gripper-servo-mode=learned ^
  --gripper-learned-control-space=full_pose ^
  --gripper-center-tolerance-px=20 ^
  --no-duplicate-camera-auto-reopen ^
  --gripper-final-iters=5 ^
  --gripper-refine-if-visible-after-full-pose-error ^
  --gripper-refine-visible-max-joint-error-deg=2.5 ^
  --gripper-learned-autonomous-search ^
  --gripper-learned-search-step-deg=1.20 ^
  --gripper-learned-search-axis-step-deg=0.60 ^
  --no-gripper-learned-search-axis-first ^
  --gripper-learned-allow-axis-probes ^
  --gripper-learned-search-max-candidates=12 ^
  --gripper-learned-search-axis-joints shoulder_pan.pos shoulder_lift.pos elbow_flex.pos wrist_flex.pos ^
  --gripper-learned-full-min-local-samples=1 ^
  --no-gripper-learned-full-require-both-local ^
  --no-gripper-learned-full-allow-global-sample ^
  --gripper-learned-k=1 ^
  --gripper-learned-min-improve-px=1.0 ^
  --gripper-learned-accept-improve-px=0.75 ^
  --gripper-learned-neutral-band-px=1.0 ^
  --gripper-learned-reverse-worse-px=1.5 ^
  --gripper-learned-fallback-after-worse-px=0.5 ^
  --gripper-learned-rollback-worse-px=1.0 ^
  --gripper-learned-local-overhead-radius-px=80 ^
  --gripper-learned-local-joint-radius-deg=8 ^
  --gripper-learned-direction-gate-error-px=25 ^
  --gripper-learned-direction-min-cos=0.65 ^
  --gripper-learned-full-min-step-deg=1.00 ^
  --gripper-learned-full-max-step-deg=6.00 ^
  --gripper-learned-full-max-pan-deg=2.50 ^
  --gripper-learned-full-max-lift-deg=4.00 ^
  --gripper-learned-full-max-elbow-deg=4.00 ^
  --gripper-learned-full-max-wrist-flex-deg=2.50 ^
  --gripper-learned-full-max-wrist-roll-deg=2.00 ^
  --no-gripper-learned-clearance ^
  --gripper-learned-clearance-max-deg=1.00 ^
  --gripper-learned-clearance-joints shoulder_lift.pos elbow_flex.pos wrist_flex.pos ^
  --gripper-learned-near-done-px=20 ^
  --gripper-learned-refine-outer-correction ^
  --gripper-learned-settle-threshold-deg=0.35 ^
  --gripper-learned-rollback-outer-correction ^
  --no-gripper-learned-stop-after-accepted-move ^
  --gripper-learned-require-linked-sample ^
  --gripper-learned-use-param-model ^
  --gripper-learned-auto-update-params ^
  --gripper-learned-param-history=3 ^
  --gripper-learned-param-only ^
  --coarse-bias-enable ^
  --coarse-bias-auto-update ^
  --coarse-bias-history=3 ^
  --coarse-bias-max-joint-deg=4.0 ^
  --coarse-bias-min-improve-px=1.0 ^
  --gripper-learned-sample-roles pre_refine_to_manual manual_before_to_manual manual_correction ^
  --gripper-max-step-deg=0.45 ^
  --gripper-min-step-deg=0.12 ^
  --gripper-min-actual-motion-deg=0.12 ^
  --gripper-refine-joint-speed-deg-s=18 ^
  --gripper-refine-min-move-steps=8 ^
  --gripper-settle-s=0.65 ^
  --gripper-observation-samples=5 ^
  --gripper-observation-interval-s=0.04 ^
  %*

if not "%ERRORLEVEL%"=="0" exit /b %ERRORLEVEL%

endlocal
