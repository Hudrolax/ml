with open('/Users/hudro/google_drive/obsidian/Base/страны.md') as f:
    lc = f.readlines()

lc.sort()
with open('/Users/hudro/google_drive/obsidian/Base/страны.md', 'w') as f:
    f.writelines(lc)

print(lc)
