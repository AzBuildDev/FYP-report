@echo off
chcp 65001 >nul
rem Ports identified by the saved USB serial mapping on 2026-10-05.
rem Both arms and all 12 motors responded; saved calibration registers match.
set "FOLLOWER_PORT=COM13"
set "LEADER_PORT=COM12"
rem Camera indices are zero-based: 0=PC Camera (built-in),
rem 1=1080P USB Camera (desktop), 2=icspring camera (gripper).
set "OVERHEAD_CAMERA=1"
set "GRIPPER_CAMERA=2"
