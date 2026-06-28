def clean_roll(roll: str) -> str:
    if len(roll) == 4 and roll.startswith('0'):
        return roll[1:]
    return roll

print(clean_roll("0002")) # should be 002
print(clean_roll("0205")) # should be 205
print(clean_roll("1008")) # should be 1008
print(clean_roll("002"))  # should be 002
