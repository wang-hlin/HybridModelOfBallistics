# value of Cd to Mach number, change Mach number to velocity in ft/s
# input velocity, return relative Cd value
def cdg1(velocity):
    c_d = 0
    if 0 <= velocity and velocity < 562.66:  # 0-0.5
        c_d = ((0.203 - 0.263) * velocity / 562.66) + 0.263
    elif velocity <= 675.197:  # 0.5-0.6
        c_d = 0.203
    elif velocity <= 787.73:  # 0.6-0.7
        c_d = ((0.217 - 0.203) * (velocity - 675.197) / (787.73 - 675.197)) + 0.203
    elif velocity <= 900.262:  # 0.7-0.8
        c_d = ((0.255 - 0.217) * (velocity - 787.73) / (900.262 - 787.73)) + 0.217
    elif velocity <= 1012.795:  # 0.8-0.9
        c_d = ((0.342 - 0.255) * (velocity - 900.262) / (1012.795 - 900.262)) + 0.255
    elif velocity <= 1069.062:  # 0.9-0.95
        c_d = ((0.408 - 0.342) * (velocity - 1012.795) / (1069.062 - 1012.795)) + 0.342
    elif velocity <= 1125.328:  # 0.95-1
        c_d = ((0.481 - 0.408) * (velocity - 1069.062) / (1125.328 - 1069.062)) + 0.408
    elif velocity <= 1181.594:  # 1-1.05
        c_d = ((0.543 - 0.481) * (velocity - 1125.328) / (1181.594 - 1125.328)) + 0.481
    elif velocity <= 1237.861:  # 1.05-1.1
        c_d = ((0.588 - 0.543) * (velocity - 1181.594) / (1237.861 - 1181.594)) + 0.543
    elif velocity <= 1350.394:  # 1.1-1.2
        c_d = ((0.639 - 0.588) * (velocity - 1237.861) / (1350.394 - 1237.861)) + 0.588
    elif velocity <= 1462.927:  # 1.2-1.3
        c_d = ((0.659 - 0.639) * (velocity - 1350.394) / (1462.927 - 1350.394)) + 0.639
    elif velocity <= 1575.459:  # 1.3-1.4
        c_d = ((0.663 - 0.659) * (velocity - 1462.927) / (1575.459 - 1462.927)) + 0.659
    elif velocity <= 1687.992:  # 1.4-1.5
        c_d = ((0.657 - 0.663) * (velocity - 1575.459) / (1687.992 - 1575.459)) + 0.663
    elif velocity <= 1800.525:  # 1.5-1.6
        c_d = ((0.647 - 0.657) * (velocity - 1687.992) / (1800.525 - 1687.992)) + 0.657
    elif velocity <= 2025.591:  # 1.6-1.8
        c_d = ((0.621 - 0.647) * (velocity - 1800.525) / (2025.591 - 1800.525)) + 0.647
    elif velocity <= 2250.656:  # 1.8-2
        c_d = ((0.593 - 0.621) * (velocity - 2025.591) / (2250.656 - 2025.591)) + 0.621
    elif velocity <= 2475.722:  # 2-2.2
        c_d = ((0.569 - 0.593) * (velocity - 2250.656) / (2475.722 - 2250.656)) + 0.593
    elif velocity <= 2813.320:  # 2.2-2.5
        c_d = ((0.540 - 0.569) * (velocity - 2475.722) / (2813.320 - 2475.722)) + 0.569
    elif velocity <= 3375.984:  # 2.5-3
        c_d = ((0.513 - 0.540) * (velocity - 2813.320) / (3375.984 - 2813.320)) + 0.540
    elif velocity <= 3938.648:  # 3-3.5
        c_d = ((0.504 - 0.513) * (velocity - 3375.984) / (3938.648 - 3375.984)) + 0.513
    elif velocity <= 4501.312:  # 3.5-4
        c_d = ((0.501 - 0.504) * (velocity - 3938.648) / (4501.312 - 3938.648)) + 0.504

    return c_d