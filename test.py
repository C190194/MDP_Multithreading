s = "ob: 2, c: 4, r: 7"

# print(s.split([",", ": "]))

for cmd_tuple in complete_path:
    obs_id = cmd_tuple[0]
    cmd_list = cmd_tuple[1]
    cmd_str = "".join(cmd_list)
    