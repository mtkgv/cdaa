import os
import shutil

exit(1) # Backstop for accidental use

for file_name in os.listdir("."):

    if "verify" in file_name or "new" in file_name or file_name == "strip_comma.py":
        continue

    with open(file_name, "r+") as old_file:
        with open("new_" + file_name, "w") as new_file:
            for line in old_file:
                new_line = line.strip(",\n") + "\n"
                new_file.write(new_line)

    

    shutil.move("new_" + file_name, file_name)
