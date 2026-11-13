@echo off
chcp 65001 >nul
rem Ports verified by USB serial and read-only motor checks on 2026-10-24.
rem Leader USB serial 5C82108314; follower USB serial 5C82108242.
set "FOLLOWER_PORT=COM4"
set "LEADER_PORT=COM3"
rem Camera indices are zero-based: 0=PC Camera (built-in),
rem 1=1080P USB Camera (desktop), 2=icspring camera (gripper).
set "OVERHEAD_CAMERA=1"
set "GRIPPER_CAMERA=2"
