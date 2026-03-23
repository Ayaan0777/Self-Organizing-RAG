import os
files = []
for root, dirs, filenames in os.walk(r'C:\Users\hegde\Downloads'):
    for f in filenames:
        if f.endswith('.pdf'):
            files.append(os.path.join(root, f))
            if len(files) > 10: break
    if len(files) > 10: break

with open('files_utf8.txt', 'w', encoding='utf-8') as out:
    for f in files:
        out.write(f + '\n')
