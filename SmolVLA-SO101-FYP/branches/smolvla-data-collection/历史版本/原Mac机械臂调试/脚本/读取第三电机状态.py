#!/usr/bin/env python3
"""Read follower motor ID 3 status after a suspected torque drop."""

from __future__ import annotations

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


FOLLOWER_PORT = "/dev/tty.usbmodem5C821082421"
MOTOR_NAME = "elbow_flex"


def read_register(bus: FeetechMotorsBus, register: str) -> str:
    try:
        value = bus.read(register, MOTOR_NAME, normalize=False, num_retry=2)
        return str(value)
    except Exception as exc:
        return f"ERROR {type(exc).__name__}: {exc}"


def main() -> None:
    bus = FeetechMotorsBus(
        port=FOLLOWER_PORT,
        motors={MOTOR_NAME: Motor(3, "sts3215", MotorNormMode.RANGE_M100_100)},
    )
    registers = [
        "Model_Number",
        "Torque_Enable",
        "Present_Position",
        "Goal_Position",
        "Present_Load",
        "Present_Current",
        "Present_Voltage",
        "Present_Temperature",
        "Status",
        "Moving",
        "P_Coefficient",
        "I_Coefficient",
        "D_Coefficient",
        "Acceleration",
        "Maximum_Acceleration",
        "Max_Torque_Limit",
        "Torque_Limit",
        "Protection_Current",
        "Protective_Torque",
        "Protection_Time",
        "Overload_Torque",
        "Unloading_Condition",
        "LED_Alarm_Condition",
    ]

    try:
        bus.connect(handshake=False)
        print(f"ping id3: {bus.ping(MOTOR_NAME, num_retry=2, raise_on_error=False)}")
        for register in registers:
            print(f"{register}: {read_register(bus, register)}")
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
